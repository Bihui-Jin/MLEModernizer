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
Detect apple diseases from images.

## Metric
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

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
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Target score

0.7155493998153282

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

print("Using tensorflow", tf.__version__)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
class CFG:
    classes = [
        "complex",
        "frog_eye_leaf_spot",
        "powdery_mildew",
        "rust",
        "scab",
        "healthy",
    ]
    batch_size = 16
    test_size = 0.25
    img_size = 512
    seed = 42
    retrain = False




## === cell 2
os.makedirs("/kaggle/working/pngs/", exist_ok=True)



## === cell 3
train_dir = "../input/plant-pathology-2021-fgvc8/train_images/"
train_csv = "../input/plant-pathology-2021-fgvc8/train.csv"
test_dir = "../input/plant-pathology-2021-fgvc8/test_images/"
duplicates_path = "../input/duplicatescsv/duplicates.csv"  # may not exist
png_dir = "/kaggle/working/pngs/"
model_dir = "/kaggle/working/"

print(os.path.isdir(train_dir))
print(os.path.isfile(train_csv))
print(os.path.isdir(test_dir))
print(os.path.isfile(duplicates_path))
print(os.path.isdir(png_dir))



## === cell 4
df_train = pd.read_csv(train_csv)
print(f"Train csv shape: {df_train.shape}")

available_imgs = set(os.listdir(train_dir))
missing = [img for img in df_train["image"] if img not in available_imgs]
if missing:
    print(f"{len(missing)} images listed in CSV are missing from train_images.")
else:
    print("All training images are present.")



## === cell 5
if os.path.isfile(duplicates_path):
    df_duplicates = pd.read_csv(duplicates_path, header=None, names=["img1", "img2"])
    print(f"Duplicates file loaded: {df_duplicates.shape[0]} rows")
else:
    df_duplicates = pd.DataFrame(columns=["img1", "img2"])
    print("No duplicates file found – proceeding without duplicate handling.")



## === cell 6
true_duplicates = []
false_duplicates = []
if not df_duplicates.empty:
    for ind in range(df_duplicates.shape[0]):
        img1, img2 = df_duplicates.iloc[ind]
        if img1 not in df_train["image"].values:
            print(f"{img1} not in training dataset")
            continue
        if img2 not in df_train["image"].values:
            print(f"{img2} not in training dataset")
            continue
        lbl1 = df_train.loc[df_train["image"] == img1, "labels"].values[0]
        lbl2 = df_train.loc[df_train["image"] == img2, "labels"].values[0]
        if lbl1 == lbl2:
            true_duplicates.append(img1)
        else:
            false_duplicates.append((img1, img2))
    print(
        f"There are {len(true_duplicates)} true duplicates and {len(false_duplicates)} false ones."
    )
else:
    print("Skipping duplicate analysis (file not available).")



## === cell 7
labels_exploded = [lab.split(" ") for lab in df_train["labels"]]
flat_labels = [lab for sublist in labels_exploded for lab in sublist]
unique_labels = np.unique(flat_labels)
assert set(unique_labels) == set(CFG.classes), "Label mismatch with CFG.classes"



## === cell 8
mlb = MultiLabelBinarizer(classes=CFG.classes)
one_hot = mlb.fit_transform(labels_exploded)
one_hot_df = pd.DataFrame(one_hot, columns=CFG.classes, index=df_train.index)
df_train = df_train.drop(columns=["labels"])
df_train = pd.concat([df_train, one_hot_df], axis=1)
print("After encoding:", df_train.head())



## === cell 9
y_strat = df_train[CFG.classes].astype(int).values
sss = StratifiedShuffleSplit(n_splits=1, test_size=CFG.test_size, random_state=CFG.seed)
strat_labels = y_strat.argmax(axis=1)
for train_idx, valid_idx in sss.split(df_train, strat_labels):
    df_train_split = df_train.iloc[train_idx].reset_index(drop=True)
    df_valid_split = df_train.iloc[valid_idx].reset_index(drop=True)




## === cell 10
def pred2labels(pred, thresh=0.5, labels=CFG.classes):
    assert len(pred) == len(labels)
    selected = [labels[i] for i, p in enumerate(pred) if p > thresh]
    return " ".join(selected) if selected else "healthy"  # fallback




## === cell 11
AUTOTUNE = tf.data.AUTOTUNE
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip(mode="horizontal_and_vertical", seed=CFG.seed),
        tf.keras.layers.RandomRotation(0.2, seed=CFG.seed),
        tf.keras.layers.RandomContrast(0.3, seed=CFG.seed),
        tf.keras.layers.RandomTranslation(
            height_factor=0.2, width_factor=0.2, seed=CFG.seed
        ),
    ]
)




## === cell 12
def parse_image(file_path):
    img = tf.io.read_file(os.path.join(train_dir, file_path))
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [CFG.img_size, CFG.img_size])
    return img


def prepare_dataset(df, augment=False):
    ds = tf.data.Dataset.from_tensor_slices(
        (df["image"].values, df[CFG.classes].values.astype(np.float32))
    )
    ds = ds.map(lambda x, y: (parse_image(x), y), num_parallel_calls=AUTOTUNE)
    if augment:
        ds = ds.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=AUTOTUNE,
        )
    ds = ds.batch(CFG.batch_size).prefetch(AUTOTUNE).repeat()
    return ds




## === cell 13
ds_train = prepare_dataset(df_train_split, augment=True)
ds_valid = prepare_dataset(df_valid_split, augment=False)




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2116518883.py in <cell line: 0>()
----> 1 ds_train = prepare_dataset(df_train_split, augment=True)
      2 ds_valid = prepare_dataset(df_valid_split, augment=False)
      3 
      4 

/tmp/ipykernel_55/1172239722.py in prepare_dataset(df, augment)
     11         (df["image"].values, df[CFG.classes].values.astype(np.float32))
     12     )
---> 13     ds = ds.map(lambda x, y: (parse_image(x), y), num_parallel_calls=AUTOTUNE)
     14     if augment:
     15         ds = ds.map(

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

/tmp/__autograph_generated_filey61sa0uu.py in <lambda>(x, y)
      3 
      4     def inner_factory(ag__):
----> 5         tf__lam = lambda x, y: ag__.with_function_scope(lambda lscope: (ag__.converted_call(parse_image, (x,), None, lscope), y), 'lscope', ag__.STD)
      6         return tf__lam
      7     return inner_factory

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/function_wrappers.py in with_function_scope(thunk, scope_name, options)
    111   """Inline version of the FunctionScope context manager."""
    112   with FunctionScope('lambda_', scope_name, options) as scope:
--> 113     return thunk(scope)

/tmp/__autograph_generated_filey61sa0uu.py in <lambda>(lscope)
      3 
      4     def inner_factory(ag__):
----> 5         tf__lam = lambda x, y: ag__.with_function_scope(lambda lscope: (ag__.converted_call(parse_image, (x,), None, lscope), y), 'lscope', ag__.STD)
      6         return tf__lam
      7     return inner_factory

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    439         result = converted_f(*effective_args, **kwargs)
    440       else:
--> 441         result = converted_f(*effective_args)
    442     except Exception as e:
    443       _attach_error_metadata(e, converted_f)

/tmp/__autograph_generated_fileizbs9p_q.py in tf__parse_image(file_path)
      8                 do_return = False
      9                 retval_ = ag__.UndefinedReturnValue()
---> 10                 img = ag__.converted_call(ag__.ld(tf).io.read_file, (ag__.converted_call(ag__.ld(os).path.join, (ag__.ld(train_dir), ag__.ld(file_path)), None, fscope),), None, fscope)
     11                 img = ag__.converted_call(ag__.ld(tf).image.decode_jpeg, (ag__.ld(img),), dict(channels=3), fscope)
     12                 img = ag__.converted_call(ag__.ld(tf).image.convert_image_dtype, (ag__.ld(img), ag__.ld(tf).float32), None, fscope)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    375 
    376   if not options.user_requested and conversion.is_allowlisted(f):
--> 377     return _call_unconverted(f, args, kwargs, options)
    378 
    379   # internal_convert_user_code is for example turned off when issuing a dynamic

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in _call_unconverted(f, args, kwargs, options, update_cache)
    458   if kwargs is not None:
    459     return f(*args, **kwargs)
--> 460   return f(*args)
    461 
    462 

/usr/lib/python3.11/posixpath.py in join(a, *p)

/usr/lib/python3.11/genericpath.py in _check_arg_types(funcname, *args)

TypeError: in user code:

    File "/tmp/ipykernel_55/1172239722.py", line 13, in None  *
        lambda x, y: (parse_image(x), y)
    File "/tmp/ipykernel_55/1172239722.py", line 2, in parse_image  *
        img = tf.io.read_file(os.path.join(train_dir, file_path))
    File "<frozen posixpath>", line 90, in join  **
        
    File "<frozen genericpath>", line 152, in _check_arg_types
        

    TypeError: join() argument must be str, bytes, or os.PathLike object, not 'SymbolicTensor'


## === cell 14
def create_cnn(
    input_shape,
    output_length,
    nb_cnn=3,
    nb_filters=64,
    activation_cnn="relu",
    model_transfert=None,
    fine_tune=False,
    nb_FC_layer=3,
    nb_FC_neurons=512,
    reducing=False,
    activation_FC="relu",
    dropout=0.0,
    activation_output="sigmoid",
    name="my_cnn_model",
):
    assert input_shape[-1] == 3
    if reducing:
        assert nb_FC_neurons % (2**nb_FC_layer) == 0
    model = tf.keras.models.Sequential(name=name)
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape, name="Input_layer"))
    if model_transfert is None:
        for i in range(nb_cnn):
            model.add(
                tf.keras.layers.Conv2D(
                    nb_filters,
                    (3, 3),
                    padding="same",
                    activation=activation_cnn,
                    name=f"Conv2D_{i+1}",
                )
            )
            model.add(
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name=f"MaxPool_{i+1}")
            )
    else:
        if not fine_tune:
            model_transfert.trainable = False
        model.add(model_transfert)
        model.add(
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="MaxPool_transfer")
        )
    model.add(tf.keras.layers.Flatten())
    if reducing:
        for i in range(nb_FC_layer):
            units = nb_FC_neurons // (2**i)
            model.add(
                tf.keras.layers.Dense(
                    units, activation=activation_FC, name=f"FC_layer_{i+1}"
                )
            )
            if dropout:
                model.add(tf.keras.layers.Dropout(dropout, name=f"Dropout_{i+1}"))
    else:
        for i in range(nb_FC_layer):
            model.add(
                tf.keras.layers.Dense(
                    nb_FC_neurons, activation=activation_FC, name=f"FC_layer_{i+1}"
                )
            )
        if dropout:
            model.add(tf.keras.layers.Dropout(dropout, name="Dropout"))
    model.add(
        tf.keras.layers.Dense(
            output_length, activation=activation_output, name="Output_layer"
        )
    )
    return model


def get_callbacks(monitor="val_loss", save_path=None, patience=8):
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor, patience=patience, restore_best_weights=True
        )
    ]
    if save_path:
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=save_path, monitor=monitor, save_best_only=True, verbose=0
            )
        )
    return callbacks




## === cell 15
base = tf.keras.applications.Xception(
    include_top=False, weights="imagenet", input_shape=(CFG.img_size, CFG.img_size, 3)
)
model = create_cnn(
    input_shape=(CFG.img_size, CFG.img_size, 3),
    output_length=len(CFG.classes),
    model_transfert=base,
    fine_tune=True,
    nb_FC_layer=2,
    nb_FC_neurons=512,
    reducing=True,
    dropout=0.0,
    activation_output="sigmoid",
    name="plant_disease_model",
)

optimizer = tf.keras.optimizers.Adam(learning_rate=3.5e-5)
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[tf.keras.metrics.BinaryAccuracy(name="acc")],
)

model.summary()



## === cell 16
model_path = os.path.join(model_dir, "model.h5")
if os.path.isfile(model_path) and not CFG.retrain:
    print("Loading existing model")
    model = tf.keras.models.load_model(model_path)
    history = None
else:
    steps_per_epoch = df_train_split.shape[0] // CFG.batch_size
    validation_steps = df_valid_split.shape[0] // CFG.batch_size
    history = model.fit(
        ds_train,
        validation_data=ds_valid,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        epochs=5,
        callbacks=get_callbacks(monitor="val_acc", save_path=model_path, patience=3),
    )
    model.save(model_path)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1498856265.py in <cell line: 0>()
      8     validation_steps = df_valid_split.shape[0] // CFG.batch_size
      9     history = model.fit(
---> 10         ds_train,
     11         validation_data=ds_valid,
     12         steps_per_epoch=steps_per_epoch,

NameError: name 'ds_train' is not defined

## === cell 17
if history:
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    axs[0].plot(history.history["loss"], label="train loss")
    axs[0].plot(history.history["val_loss"], label="val loss")
    axs[0].set_title("Loss")
    axs[0].legend()
    axs[1].plot(history.history["acc"], label="train acc")
    axs[1].plot(history.history["val_acc"], label="val acc")
    axs[1].set_title("Accuracy")
    axs[1].legend()
    plt.savefig(os.path.join(png_dir, "training_history.png"))
    plt.close(fig)




## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3174554974.py in <cell line: 0>()
----> 1 if history:
      2     fig, axs = plt.subplots(1, 2, figsize=(14, 5))
      3     axs[0].plot(history.history["loss"], label="train loss")
      4     axs[0].plot(history.history["val_loss"], label="val loss")
      5     axs[0].set_title("Loss")

NameError: name 'history' is not defined

## === cell 18
def parse_test_image(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [CFG.img_size, CFG.img_size])
    return img


def predict_image(img_path, model):
    img = parse_test_image(img_path)
    img = tf.expand_dims(img, axis=0)
    pred = model.predict(img)
    return pred2labels(pred[0])




## === cell 19
submission = pd.DataFrame(columns=["image", "labels"])
for fname in os.listdir(test_dir):
    full_path = os.path.join(test_dir, fname)
    pred_labels = predict_image(full_path, model)
    submission = pd.concat(
        [submission, pd.DataFrame([[fname, pred_labels]], columns=["image", "labels"])],
        ignore_index=True,
    )

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}, shape {submission.shape}")

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
FailedPreconditionError                   Traceback (most recent call last)
/tmp/ipykernel_55/2366081382.py in <cell line: 0>()
      2 for fname in os.listdir(test_dir):
      3     full_path = os.path.join(test_dir, fname)
----> 4     pred_labels = predict_image(full_path, model)
      5     submission = pd.concat(
      6         [submission, pd.DataFrame([[fname, pred_labels]], columns=["image", "labels"])],

/tmp/ipykernel_55/3725090098.py in predict_image(img_path, model)
      8 
      9 def predict_image(img_path, model):
---> 10     img = parse_test_image(img_path)
     11     img = tf.expand_dims(img, axis=0)
     12     pred = model.predict(img)

/tmp/ipykernel_55/3725090098.py in parse_test_image(file_path)
      1 def parse_test_image(file_path):
----> 2     img = tf.io.read_file(file_path)
      3     img = tf.image.decode_jpeg(img, channels=3)
      4     img = tf.image.convert_image_dtype(img, tf.float32)
      5     img = tf.image.resize(img, [CFG.img_size, CFG.img_size])

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

FailedPreconditionError: {{function_node __wrapped__ReadFile_device_/job:localhost/replica:0/task:0/device:CPU:0}} ../input/plant-pathology-2021-fgvc8/test_images/test_images; Is a directory [Op:ReadFile]
