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

No external packages required in the script and installed.

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

0.7764358264081261

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.24507) has done: 'The update wraps TensorFlow imports and model loading in safe try/except blocks, falling back to a simple “healthy” prediction when TensorFlow isn’t available or the model can’t be loaded. This eliminates the import and loading errors, ensures `predictions` is always defined, and writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.38173) has done: 'The changes guard TensorFlow‑related imports and generator creation so the notebook runs even when TensorFlow cannot be loaded, and replace the fallback “healthy” prediction with a more informed default based on the most common labels in the training set, which should raise the F1 score toward the target.'
- What this solution (achieved 0.28656) has done: 'We replace the previous “most‑common three labels” fallback with a single most‑common whole label string from the training data, which matches the required submission format and gives a more realistic baseline. The code now computes this default label and uses it for every test image when the TensorFlow model cannot be loaded, ensuring a valid CSV is always written. This change fixes the formatting issue that hurt the F1 score and brings the result closer to the target while keeping the original logic untouched.'
- What this solution (achieved 0.38173) has done: 'The change updates the fallback prediction: instead of using the single most frequent full label string, it now computes the three most common individual disease tokens across the training set and joins them with spaces. This richer default better reflects the label distribution, raising the mean F1 score while keeping the original pipeline unchanged.'
- What this solution (achieved 0.28656) has done: 'Implemented a safer fallback prediction by using the most frequent full label string from the training data instead of the three most common individual tokens. This aligns the default prediction with actual training label distribution, improving expected F1‑score while keeping the overall pipeline unchanged and ensuring a valid `submission.csv` is always written.'
- What this solution (achieved 0.38173) has done: 'The fix adds the missing pandas import, loads the list of test image IDs from the provided sample submission, and removes references to undefined variables. It then always uses the previously computed `default_prediction` (the three most common disease tokens) for every test image, guaranteeing a valid `submission.csv` with the correct columns. No core modeling logic is altered.'
- What this solution (achieved 0.64139) has done: 'The modification adds dataset caching to the training and validation pipelines, eliminating repeated image decoding and resizing across epochs. By calling `.cache()` after the map step, each image is processed only once, dramatically reducing I/O and preprocessing time while keeping the model architecture, training loops, and prediction logic unchanged. This speeds up execution enough to stay within the 600‑second limit without affecting result accuracy.'

# 9. Code solution

## === cell 0
try:
    import google.protobuf.message_factory as _mf

    if not hasattr(_mf.MessageFactory, "GetPrototype"):

        def _GetPrototype(self, descriptor):
            return self.pool.GetPrototype(descriptor)

        _mf.MessageFactory.GetPrototype = _GetPrototype  # type: ignore
except Exception:
    pass

try:
    import tensorflow as tf

    print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))
except Exception as e:
    tf = None
    print("TensorFlow import failed:", e)




## === cell 1
import os
import pandas as pd
import numpy as np
from collections import Counter

train_path = "/kaggle/input/plant-pathology-2021-fgvc8/train.csv"
train_df = pd.read_csv(train_path)

all_tokens = []
for lbl in train_df["labels"].astype(str):
    all_tokens.extend(lbl.split())
token_counts = Counter(all_tokens)
most_common_tokens = [tok for tok, _ in token_counts.most_common(3)]
default_prediction = " ".join(most_common_tokens)  # used when model is unavailable

unique_tokens = sorted(token_counts.keys())
token_to_idx = {tok: i for i, tok in enumerate(unique_tokens)}
num_classes = len(unique_tokens)




## === cell 2
train_images_dir = "/kaggle/input/plant-pathology-2021-fgvc8/train_images"
train_df["image_path"] = train_df["image"].apply(
    lambda x: os.path.join(train_images_dir, x)
)


def multilabel_vector(label_str):
    vec = np.zeros(num_classes, dtype=np.float32)
    for tok in label_str.split():
        idx = token_to_idx.get(tok)
        if idx is not None:
            vec[idx] = 1.0
    return vec


train_df["label_vec"] = train_df["labels"].astype(str).apply(multilabel_vector)

train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
split_idx = int(0.8 * len(train_df))
df_train = train_df.iloc[:split_idx]
df_val = train_df.iloc[split_idx:]




## === cell 3
if tf is not None:
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 64

    cache_dir = "/tmp/tf_cache"
    os.makedirs(cache_dir, exist_ok=True)
    train_cache_path = os.path.join(cache_dir, "train_cache")
    val_cache_path = os.path.join(cache_dir, "val_cache")
    train_feat_cache = os.path.join(cache_dir, "train_feat_cache")
    val_feat_cache = os.path.join(cache_dir, "val_feat_cache")

    def load_and_preprocess(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = img / 255.0  # normalize to [0,1]
        return img, label

    train_ds_raw = (
        tf.data.Dataset.from_tensor_slices(
            (df_train["image_path"].values, np.stack(df_train["label_vec"].values))
        )
        .map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
        .shuffle(1024)
        .cache(train_cache_path)
        .prefetch(AUTOTUNE)
    )
    val_ds_raw = (
        tf.data.Dataset.from_tensor_slices(
            (df_val["image_path"].values, np.stack(df_val["label_vec"].values))
        )
        .map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
        .cache(val_cache_path)
        .prefetch(AUTOTUNE)
    )

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet"
    )
    base_model.trainable = False  # freeze backbone

    def global_avg(x):
        return tf.reduce_mean(x, axis=[1, 2])

    train_feat_ds = (
        train_ds_raw.map(
            lambda img, lbl: (global_avg(base_model(img, training=False)), lbl),
            num_parallel_calls=AUTOTUNE,
        )
        .cache(train_feat_cache)
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )
    val_feat_ds = (
        val_ds_raw.map(
            lambda img, lbl: (global_avg(base_model(img, training=False)), lbl),
            num_parallel_calls=AUTOTUNE,
        )
        .cache(val_feat_cache)
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )

    feature_dim = base_model.output_shape[-1]  # 1280 for MobileNetV2
    top_model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(feature_dim,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(num_classes, activation="sigmoid"),
        ]
    )

    top_model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")],
    )

    top_model.fit(train_feat_ds, validation_data=val_feat_ds, epochs=3, verbose=2)

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs)

    model.layers[-1].set_weights(top_model.layers[-1].get_weights())
else:
    model = None  # will trigger fallback later




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/90589382.py in <cell line: 0>()
     49     # Feature extraction pipelines (cached to disk)
     50     train_feat_ds = (
---> 51         train_ds_raw.map(
     52             lambda img, lbl: (global_avg(base_model(img, training=False)), lbl),
     53             num_parallel_calls=AUTOTUNE,

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

/tmp/__autograph_generated_filegzhhodxn.py in <lambda>(img, lbl)
      3 
      4     def inner_factory(ag__):
----> 5         tf__lam = lambda img, lbl: ag__.with_function_scope(lambda lscope: (ag__.converted_call(global_avg, (ag__.converted_call(base_model, (img,), dict(training=False), lscope),), None, lscope), lbl), 'lscope', ag__.STD)
      6         return tf__lam
      7     return inner_factory

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/function_wrappers.py in with_function_scope(thunk, scope_name, options)
    111   """Inline version of the FunctionScope context manager."""
    112   with FunctionScope('lambda_', scope_name, options) as scope:
--> 113     return thunk(scope)

/tmp/__autograph_generated_filegzhhodxn.py in <lambda>(lscope)
      3 
      4     def inner_factory(ag__):
----> 5         tf__lam = lambda img, lbl: ag__.with_function_scope(lambda lscope: (ag__.converted_call(global_avg, (ag__.converted_call(base_model, (img,), dict(training=False), lscope),), None, lscope), lbl), 'lscope', ag__.STD)
      6         return tf__lam
      7     return inner_factory

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    375 
    376   if not options.user_requested and conversion.is_allowlisted(f):
--> 377     return _call_unconverted(f, args, kwargs, options)
    378 
    379   # internal_convert_user_code is for example turned off when issuing a dynamic

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in _call_unconverted(f, args, kwargs, options, update_cache)
    457 
    458   if kwargs is not None:
--> 459     return f(*args, **kwargs)
    460   return f(*args)
    461 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/layers/input_spec.py in assert_input_compatibility(input_spec, inputs, layer_name)
    243                 if spec_dim is not None and dim is not None:
    244                     if spec_dim != dim:
--> 245                         raise ValueError(
    246                             f'Input {input_index} of layer "{layer_name}" is '
    247                             "incompatible with the layer: "

ValueError: in user code:

    File "/tmp/ipykernel_11/90589382.py", line 52, in None  *
        lambda img, lbl: (global_avg(base_model(img, training=False)), lbl)
    File "/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py", line 122, in error_handler  **
        raise e.with_traceback(filtered_tb) from None
    File "/usr/local/lib/python3.11/dist-packages/keras/src/layers/input_spec.py", line 245, in assert_input_compatibility
        raise ValueError(

    ValueError: Input 0 of layer "mobilenetv2_1.00_224" is incompatible with the layer: expected shape=(None, 224, 224, 3), found shape=(224, 224, 3)


## === cell 4
best_thresh = 0.5
if tf is not None and model is not None:
    from sklearn.metrics import f1_score

    val_probs = top_model.predict(val_feat_ds, verbose=0)
    y_true = np.stack(df_val["label_vec"].values)

    thresholds = np.arange(0.30, 0.71, 0.05)
    best_f1 = 0.0
    for thr in thresholds:
        y_pred = (val_probs > thr).astype(int)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thr
    print(f"Best validation macro‑F1 {best_f1:.4f} at threshold {best_thresh:.2f}")




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1196232339.py in <cell line: 0>()
      1 best_thresh = 0.5
----> 2 if tf is not None and model is not None:
      3     from sklearn.metrics import f1_score
      4 
      5     # Use the lightweight top model on cached validation features

NameError: name 'model' is not defined

## === cell 5
sample_sub_path = "/kaggle/input/plant-pathology-2021-fgvc8/sample_submission.csv"
test_df = pd.read_csv(sample_sub_path)
test_ids = test_df["image"].astype(str).tolist()
test_images_dir = "/kaggle/input/plant-pathology-2021-fgvc8/test_images"
test_paths = [os.path.join(test_images_dir, img_id) for img_id in test_ids]

predictions = []

if tf is not None and model is not None:

    def load_test(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = img / 255.0
        return img

    test_ds_raw = tf.data.Dataset.from_tensor_slices(test_paths)
    test_ds_raw = test_ds_raw.map(load_test, num_parallel_calls=AUTOTUNE)

    test_feat_ds = (
        test_ds_raw.map(
            lambda img: global_avg(base_model(img, training=False)),
            num_parallel_calls=AUTOTUNE,
        )
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )

    prob_arrays = top_model.predict(test_feat_ds, verbose=0)
    for probs in prob_arrays:
        token_idxs = np.where(probs > best_thresh)[0]
        if len(token_idxs) == 0:
            predictions.append(default_prediction)
        else:
            tokens = [unique_tokens[i] for i in token_idxs]
            predictions.append(" ".join(tokens))
else:
    predictions = [default_prediction] * len(test_ids)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/874063952.py in <cell line: 0>()
      7 predictions = []
      8 
----> 9 if tf is not None and model is not None:
     10 
     11     def load_test(path):

NameError: name 'model' is not defined

## === cell 6
submission = pd.DataFrame({"image": test_ids, "labels": predictions})
submission.to_csv("submission.csv", index=False)

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1011525481.py in <cell line: 0>()
----> 1 submission = pd.DataFrame({"image": test_ids, "labels": predictions})
      2 submission.to_csv("submission.csv", index=False)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __init__(self, data, index, columns, dtype, copy)
    776         elif isinstance(data, dict):
    777             # GH#38939 de facto copy defaults to False only in non-dict cases
--> 778             mgr = dict_to_mgr(data, index, columns, dtype=dtype, copy=copy, typ=manager)
    779         elif isinstance(data, ma.MaskedArray):
    780             from numpy.ma import mrecords

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in dict_to_mgr(data, index, columns, dtype, typ, copy)
    501             arrays = [x.copy() if hasattr(x, "dtype") else x for x in arrays]
    502 
--> 503     return arrays_to_mgr(arrays, columns, index, dtype=dtype, typ=typ, consolidate=copy)
    504 
    505 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in arrays_to_mgr(arrays, columns, index, dtype, verify_integrity, typ, consolidate)
    112         # figure out the index, if necessary
    113         if index is None:
--> 114             index = _extract_index(arrays)
    115         else:
    116             index = ensure_index(index)

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in _extract_index(data)
    675         lengths = list(set(raw_lengths))
    676         if len(lengths) > 1:
--> 677             raise ValueError("All arrays must be of the same length")
    678 
    679         if have_dicts:

ValueError: All arrays must be of the same length
