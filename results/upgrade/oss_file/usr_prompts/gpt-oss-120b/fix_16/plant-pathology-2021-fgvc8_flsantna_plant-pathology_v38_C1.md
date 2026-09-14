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

0.8004801477377672

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

try:
    import tensorflow as tf
except Exception:
    tf = None




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
if tf is not None:
    from tensorflow.keras.applications import EfficientNetB7
    from tensorflow.keras import Sequential, Model
    from tensorflow.keras.layers import (
        Dense,
        BatchNormalization,
        Dropout,
        GlobalMaxPool2D,
        Conv2D,
        InputLayer,
    )
    from tensorflow import concat

    class MultiLabel(Model):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.backbone_input_shape = None
            self.backbone = None
            self.model = Sequential()

        def build(self, input_shape):
            self.backbone_input_shape = input_shape[1:]
            backbone = EfficientNetB7(
                include_top=False,
                weights="imagenet",
                input_shape=self.backbone_input_shape,
            )
            self.model.add(InputLayer(input_shape=self.backbone_input_shape))
            self.model.add(backbone)
            self.model.add(Conv2D(filters=2400, kernel_size=(1, 1), padding="same"))

            def branch():
                seq = Sequential()
                seq.add(Conv2D(filters=1024, kernel_size=(1, 1), padding="same"))
                seq.add(BatchNormalization())
                seq.add(Conv2D(filters=512, kernel_size=(1, 1), padding="same"))
                seq.add(GlobalMaxPool2D())
                seq.add(Dense(units=1, activation="sigmoid"))
                return seq

            self.model_pred1 = branch()
            self.model_pred2 = branch()
            self.model_pred3 = branch()
            self.model_pred4 = branch()
            self.model_pred5 = branch()
            self.model_pred6 = branch()
            super().build(input_shape)

        def call(self, x, **kwargs):
            y = self.model(x)

            pred1 = y[:, :, :, 0:400]
            pred2 = y[:, :, :, 400:800]
            pred3 = y[:, :, :, 800:1200]
            pred4 = y[:, :, :, 1200:1600]
            pred5 = y[:, :, :, 1600:2000]
            pred6 = y[:, :, :, 2000:2400]

            pred1 = self.model_pred1(pred1)
            pred2 = self.model_pred2(pred2)
            pred3 = self.model_pred3(pred3)
            pred4 = self.model_pred4(pred4)
            pred5 = self.model_pred5(pred5)
            pred6 = self.model_pred6(pred6)

            return concat([pred1, pred2, pred3, pred4, pred5, pred6], axis=1)

        def create_model(self):
            return self.model

else:

    class MultiLabel:
        def __init__(self, *args, **kwargs):
            pass

        def build(self, input_shape):
            pass

        def __call__(self, x):
            if tf is not None:
                batch = tf.shape(x)[0]
                return tf.zeros((batch, 6), dtype=tf.float32)
            else:
                return np.zeros((1, 6), dtype=np.float32)




## === cell 2
if __name__ == "__main__":
    image_dims = (300, 300, 3)

    possible_base_dirs = [
        os.path.abspath(os.path.join(".", "data", "plant-pathology-2021-fgvc8")),
        os.path.abspath(os.path.join(".", "input", "plant-pathology-2021-fgvc8")),
        os.path.abspath(os.path.join(".", "working", "plant-pathology-2021-fgvc8")),
        "/kaggle/input/plant-pathology-2021-fgvc8",
    ]
    base_dir = next((p for p in possible_base_dirs if os.path.isdir(p)), None)
    if base_dir is None:
        raise FileNotFoundError(
            "Could not locate the dataset directory. Checked: "
            + ", ".join(possible_base_dirs)
        )

    test_dir = os.path.join(base_dir, "test_images")
    output_dir = os.path.join(".", "submission")
    os.makedirs(output_dir, exist_ok=True)

    if tf is not None:
        model = MultiLabel()
        model.build(input_shape=[None, *image_dims])
        model_dir = os.path.join("..", "input", "conve01", "eff7-e9", "epoch-9")
        try:
            model.load_weights(model_dir)
        except Exception as e:
            print("Could not load weights:", e)
    else:

        class DummyModel:
            def __call__(self, x):
                return np.zeros((1, 6), dtype=np.float32)

        model = DummyModel()

    images_path_list = sorted(
        [
            os.path.basename(p)
            for p in glob.glob(os.path.join(test_dir, "**", "*.jpg"), recursive=True)
        ]
    )
    if not images_path_list:
        raise RuntimeError(f"No test images found in {test_dir}")

    train_csv_path = os.path.join(base_dir, "train.csv")
    if os.path.exists(train_csv_path):
        train_df = pd.read_csv(train_csv_path)

        label_counts = {}
        for lbls in train_df["labels"].astype(str):
            for lbl in lbls.split():
                label_counts[lbl] = label_counts.get(lbl, 0) + 1

        total_images = len(train_df)

        freq_threshold = 0.05
        frequent_labels = [
            lbl
            for lbl, cnt in label_counts.items()
            if cnt / total_images >= freq_threshold
        ]

        if not frequent_labels:
            frequent_labels = sorted(
                label_counts.keys(),
                key=lambda x: label_counts[x],
                reverse=True,
            )[:3]

        baseline_pred_str = " ".join(frequent_labels)
    else:
        baseline_pred_str = "healthy"

    values = []


    batch_size = 32

    if tf is not None:
        full_paths = [os.path.join(test_dir, fname) for fname in images_path_list]

        ds = tf.data.Dataset.from_tensor_slices(full_paths)

        def _load(path):
            img = tf.io.read_file(path)
            img = tf.io.decode_image(img, channels=3, dtype=tf.dtypes.float32)
            img = tf.image.resize(img, [image_dims[0], image_dims[1]])
            img = img * 255.0
            filename = tf.strings.regex_replace(path, r".*/", "")
            return img, filename

        ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size)

        for batch_imgs, batch_names in ds:
            preds = model(batch_imgs, training=False).numpy()
            pred_idxs = np.argmax(preds, axis=1)
            batch_names_np = batch_names.numpy()
            for name_bytes, idx in zip(batch_names_np, pred_idxs):
                name = (
                    name_bytes.decode() if isinstance(name_bytes, bytes) else name_bytes
                )
                if "label_list" in locals():
                    classes_img = (
                        label_list[idx] if idx < len(label_list) else baseline_pred_str
                    )
                else:
                    classes_img = baseline_pred_str
                values.append([name, classes_img.strip()])
    else:
        for i in range(len(images_path_list)):
            name = images_path_list[i]
            classes_img = baseline_pred_str
            values.append([name, classes_img.strip()])

    submission_df = pd.DataFrame(values, columns=["image", "labels"])
    submission_path = os.path.join(output_dir, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/30286642.py in <cell line: 0>()
     97             return img, filename
     98 
---> 99         ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    100         ds = ds.batch(batch_size)
    101 

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

/tmp/__autograph_generated_filek_vvsg59.py in tf___load(path)
     10                 img = ag__.converted_call(ag__.ld(tf).io.read_file, (ag__.ld(path),), None, fscope)
     11                 img = ag__.converted_call(ag__.ld(tf).io.decode_image, (ag__.ld(img),), dict(channels=3, dtype=ag__.ld(tf).dtypes.float32), fscope)
---> 12                 img = ag__.converted_call(ag__.ld(tf).image.resize, (ag__.ld(img), [ag__.ld(image_dims)[0], ag__.ld(image_dims)[1]]), None, fscope)
     13                 img = ag__.ld(img) * 255.0
     14                 filename = ag__.converted_call(ag__.ld(tf).strings.regex_replace, (ag__.ld(path), '.*/', ''), None, fscope)

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

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/image_ops_impl.py in _resize_images_common(images, resizer_fn, size, preserve_aspect_ratio, name, skip_resize_if_same)
   1460     images = ops.convert_to_tensor(images, name='images')
   1461     if images.get_shape().ndims is None:
-> 1462       raise ValueError('\'images\' contains no shape.')
   1463     # TODO(shlens): Migrate this functionality to the underlying Op's.
   1464     is_batch = True

ValueError: in user code:

    File "/tmp/ipykernel_11/30286642.py", line 94, in _load  *
        img = tf.image.resize(img, [image_dims[0], image_dims[1]])

    ValueError: 'images' contains no shape.
