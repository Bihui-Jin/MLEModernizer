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
Given a dataset of images from digital pathology scans, predict if the center 32x32px region of a patch contains at least one pixel of tumor tissue. Tumor tissue in the outer region of the patch does not influence the label. 

## Metric
Area under the ROC curve.

## Submission Format
For each `id` in the test set, you must predict a probability that center 32x32px region of a patch contains at least one pixel of tumor tissue. The file should contain a header and have the following format:

```
id,label
0b2ea2a822ad23fdb1b5dd26653da899fbd2c0d5,0
95596b92e5066c5c52466c90b69ff089b39f2737,0
248e6738860e2ebcf6258cdc1f32f299e0c76914,0
etc.
```

## Dataset
Files are named with an image `id`. The `train_labels.csv` file provides the ground truth for the images in the `train` folder. You are predicting the labels for the images in the `test` folder.

# 2. Python version

3.12

# 3. Installed packages

geopandas==0.14.4
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
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
        input/
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
        working/
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
```

-> data/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> (stopped after 10 files for performance)

# 5. Target score

0.7613

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import time
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import tensorflow as tf
import tensorflow_io as tfio
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

from tensorflow.keras import layers, models
from sklearn.metrics import roc_auc_score
from tensorflow.keras.models import load_model

tf.random.set_seed(0)
np.random.seed(0)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.data.Options().experimental_deterministic = True
except Exception:
    pass

print("TF:", tf.__version__)
print("Num GPUs:", len(tf.config.list_physical_devices("GPU")))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
input_dir = "/kaggle/input/histopathologic-cancer-detection"

train_labels_path = os.path.join(input_dir, "train_labels.csv")
sample_sub_path = os.path.join(input_dir, "sample_submission.csv")

train_dir = os.path.join(input_dir, "train") + os.sep
test_dir = os.path.join(input_dir, "test") + os.sep

assert os.path.exists(train_labels_path), f"Missing: {train_labels_path}"
assert os.path.exists(sample_sub_path), f"Missing: {sample_sub_path}"
assert os.path.isdir(train_dir), f"Missing dir: {train_dir}"
assert os.path.isdir(test_dir), f"Missing dir: {test_dir}"

sample_data = pd.read_csv(sample_sub_path)
train_data = pd.read_csv(train_labels_path)

sample_data.head(), train_data.head(), train_dir, test_dir




## === cell 2
def print_short_summary(name, data):
    """
    Prints data head, shape and info.
    Args:
        name (str): name of dataset
        data (dataframe): dataset in a pd.DataFrame format
    """
    print(name)
    print("\n1. Data head:")
    print(data.head())
    print("\n2. Data shape: {}".format(data.shape))
    print("\n3. Data info:")
    data.info()


def print_number_files(dirpath):
    print("{}: {} files".format(dirpath, len(os.listdir(dirpath))))




## === cell 3
print_short_summary("Train data", train_data)
print_short_summary("Sample submission", sample_data)



## === cell 4
pass



## === cell 5
no_cancer = train_data[train_data["label"] == 0]
cancer = train_data[train_data["label"] == 1]

no_cancer_downsampled = resample(
    no_cancer,
    replace=False,
    n_samples=len(cancer),
    random_state=0,
)

balanced_train_data = pd.concat([no_cancer_downsampled, cancer])
balanced_train_data = balanced_train_data.sample(frac=1, random_state=0).reset_index(
    drop=True
)

balanced_train_data["label"].value_counts()



## === cell 6
image_paths = (train_dir + balanced_train_data["id"] + ".tif").values
labels = balanced_train_data["label"].values.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    image_paths,
    labels,
    test_size=0.25,
    shuffle=True,
    random_state=0,
    stratify=labels,
)

len(X_train), len(X_test)



## === cell 7
AUTOTUNE = tf.data.AUTOTUNE


def _decode_resize_rgba_tf(image_path):
    image_bytes = tf.io.read_file(image_path)

    def _decode_with_tfio(b):
        img = tfio.experimental.image.decode_tiff(b)
        img = tf.cast(img, tf.uint8)
        return img

    def _decode_with_pil(b):
        def _py_decode(x):
            import numpy as _np
            from PIL import Image as _Image
            import io as _io

            im = _Image.open(_io.BytesIO(x)).convert("RGBA")
            return _np.array(im, dtype=_np.uint8)

        img = tf.py_function(_py_decode, [b], Tout=tf.uint8)
        img.set_shape([None, None, 4])
        return img

    img = tf.cond(
        tf.constant(True),
        lambda: tf.ensure_shape(_decode_with_tfio(image_bytes), [None, None, None]),
        lambda: _decode_with_pil(image_bytes),
    )

    c = tf.shape(img)[-1]

    def _to_rgba_from_3():
        rgb = img[..., :3]
        alpha = tf.ones_like(rgb[..., :1]) * 255
        return tf.concat([rgb, alpha], axis=-1)

    def _to_rgba_from_4():
        return img[..., :4]

    def _to_rgba_from_1():
        g = img[..., :1]
        rgb = tf.concat([g, g, g], axis=-1)
        alpha = tf.ones_like(g) * 255
        return tf.concat([rgb, alpha], axis=-1)

    img = tf.case(
        [
            (tf.equal(c, 4), _to_rgba_from_4),
            (tf.equal(c, 3), _to_rgba_from_3),
            (tf.equal(c, 1), _to_rgba_from_1),
        ],
        default=_to_rgba_from_3,
        exclusive=True,
    )

    img = tf.image.resize(
        img, [32, 32], method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img = tf.cast(img, tf.float32) / 255.0
    img.set_shape([32, 32, 4])
    return img


def get_decoded_image(image_path, label=None):
    img = _decode_resize_rgba_tf(image_path)
    return img if label is None else (img, label)


def get_prefetched_data(data, batch_size, cache=True):
    opts = tf.data.Options()
    opts.experimental_deterministic = True

    if isinstance(data, (tuple, list)) and len(data) == 2:
        paths, labs = data
        dataset = tf.data.Dataset.from_tensor_slices((paths, labs))
        dataset = dataset.with_options(opts)
        dataset = dataset.map(
            get_decoded_image, num_parallel_calls=AUTOTUNE, deterministic=True
        )
    else:
        dataset = tf.data.Dataset.from_tensor_slices(data)
        dataset = dataset.with_options(opts)
        dataset = dataset.map(
            lambda p: get_decoded_image(p, None),
            num_parallel_calls=AUTOTUNE,
            deterministic=True,
        )

    if cache:
        dataset = dataset.cache()

    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset




## === cell 8
BATCH_SIZE = 128
train_dataset = get_prefetched_data((X_train, y_train), BATCH_SIZE, cache=True)
test_dataset = get_prefetched_data((X_test, y_test), BATCH_SIZE, cache=True)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NotImplementedError                       Traceback (most recent call last)
/tmp/ipykernel_11/3114980045.py in <cell line: 0>()
      1 BATCH_SIZE = 128
----> 2 train_dataset = get_prefetched_data((X_train, y_train), BATCH_SIZE, cache=True)
      3 test_dataset = get_prefetched_data((X_test, y_test), BATCH_SIZE, cache=True)
      4 
      5 

/tmp/ipykernel_11/4237036632.py in get_prefetched_data(data, batch_size, cache)
     82         dataset = tf.data.Dataset.from_tensor_slices((paths, labs))
     83         dataset = dataset.with_options(opts)
---> 84         dataset = dataset.map(
     85             get_decoded_image, num_parallel_calls=AUTOTUNE, deterministic=True
     86         )

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

/tmp/__autograph_generated_filetac1w82x.py in tf__get_decoded_image(image_path, label)
      8                 do_return = False
      9                 retval_ = ag__.UndefinedReturnValue()
---> 10                 img = ag__.converted_call(ag__.ld(_decode_resize_rgba_tf), (ag__.ld(image_path),), None, fscope)
     11                 try:
     12                     do_return = True

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    439         result = converted_f(*effective_args, **kwargs)
    440       else:
--> 441         result = converted_f(*effective_args)
    442     except Exception as e:
    443       _attach_error_metadata(e, converted_f)

/tmp/__autograph_generated_filejrh6xnj3.py in tf___decode_resize_rgba_tf(image_path)
     56                             raise
     57                         return fscope_2.ret(retval__2, do_return_2)
---> 58                 img = ag__.converted_call(ag__.ld(tf).cond, (ag__.converted_call(ag__.ld(tf).constant, (True,), None, fscope), ag__.autograph_artifact(lambda: ag__.converted_call(ag__.ld(tf).ensure_shape, (ag__.converted_call(ag__.ld(_decode_with_tfio), (ag__.ld(image_bytes),), None, fscope), [None, None, None]), None, fscope)), ag__.autograph_artifact(lambda: ag__.converted_call(ag__.ld(_decode_with_pil), (ag__.ld(image_bytes),), None, fscope))), None, fscope)
     59                 c = ag__.converted_call(ag__.ld(tf).shape, (ag__.ld(img),), None, fscope)[-1]
     60 

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

/tmp/__autograph_generated_filejrh6xnj3.py in <lambda>()
     56                             raise
     57                         return fscope_2.ret(retval__2, do_return_2)
---> 58                 img = ag__.converted_call(ag__.ld(tf).cond, (ag__.converted_call(ag__.ld(tf).constant, (True,), None, fscope), ag__.autograph_artifact(lambda: ag__.converted_call(ag__.ld(tf).ensure_shape, (ag__.converted_call(ag__.ld(_decode_with_tfio), (ag__.ld(image_bytes),), None, fscope), [None, None, None]), None, fscope)), ag__.autograph_artifact(lambda: ag__.converted_call(ag__.ld(_decode_with_pil), (ag__.ld(image_bytes),), None, fscope))), None, fscope)
     59                 c = ag__.converted_call(ag__.ld(tf).shape, (ag__.ld(img),), None, fscope)[-1]
     60 

/tmp/__autograph_generated_filejrh6xnj3.py in _decode_with_tfio(b)
     15                         do_return_1 = False
     16                         retval__1 = ag__.UndefinedReturnValue()
---> 17                         img = ag__.converted_call(ag__.ld(tfio).experimental.image.decode_tiff, (ag__.ld(b),), None, fscope_1)
     18                         img = ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(img), ag__.ld(tf).uint8), None, fscope_1)
     19                         try:

/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/experimental/image_ops.py in decode_tiff(contents, index, name)
     85       A `Tensor` of type `uint8` and shape of `[height, width, 4]` (RGBA).
     86     """
---> 87     return core_ops.io_decode_tiff(contents, index, name=name)
     88 
     89 

/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/__init__.py in __getattr__(self, attrb)
     86 
     87     def __getattr__(self, attrb):
---> 88         return getattr(self._load(), attrb)
     89 
     90     def __dir__(self):

/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/__init__.py in _load(self)
     82     def _load(self):
     83         if self._mod is None:
---> 84             self._mod = _load_library(self._library)
     85         return self._mod
     86 

/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/__init__.py in _load_library(filename, lib)
     67         except (tf.errors.NotFoundError, OSError) as e:
     68             errs.append(str(e))
---> 69     raise NotImplementedError(
     70         "unable to open file: "
     71         + f"{filename}, from paths: {filenames}\ncaused by: {errs}"

NotImplementedError: in user code:

    File "/tmp/ipykernel_11/4237036632.py", line 72, in get_decoded_image  *
        img = _decode_resize_rgba_tf(image_path)
    File "/tmp/ipykernel_11/4237036632.py", line 12, in _decode_with_tfio  *
        img = tfio.experimental.image.decode_tiff(b)
    File "/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/experimental/image_ops.py", line 87, in decode_tiff
        return core_ops.io_decode_tiff(contents, index, name=name)
    File "/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/__init__.py", line 88, in __getattr__
        return getattr(self._load(), attrb)
    File "/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/__init__.py", line 84, in _load
        self._mod = _load_library(self._library)
    File "/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/__init__.py", line 69, in _load_library
        raise NotImplementedError(

    NotImplementedError: unable to open file: libtensorflow_io.so, from paths: ['/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/libtensorflow_io.so']
    caused by: ['/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/libtensorflow_io.so: undefined symbol: _ZN3tsl8str_util9LowercaseB5cxx11ESt17basic_string_viewIcSt11char_traitsIcEE']


## === cell 9
def get_model_base():
    model = models.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 4)),
            layers.Flatten(),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    return model


def get_model_base_deep():
    model = models.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 4)),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.Flatten(),
            layers.Dense(32, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    return model


def get_model_base_wide():
    model_drop_bn = models.Sequential(
        [
            layers.Conv2D(64, (3, 3), activation="relu", input_shape=(32, 32, 4)),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    return model_drop_bn


def get_model_base_maxpool():
    model = models.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 4)),
            layers.MaxPooling2D((2, 2), strides=(2, 2)),
            layers.Flatten(),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    return model


def get_model_base_dropout():
    model = models.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 4)),
            layers.Flatten(),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.25),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    return model




## === cell 10
def get_compiled_model(func):
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        strategy = tf.distribute.MirroredStrategy()
        print("Number of devices: {}".format(strategy.num_replicas_in_sync))
    else:
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
        print("No GPU available, falling back to CPU.")

    with strategy.scope():
        compiled_model = func()
        compiled_model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.AUC(name="auc")],
        )
    return compiled_model


def plot_model_scores(scores, model_name):
    train_scores, test_scores = scores
    epochs = range(1, len(train_scores) + 1)

    plt.figure(figsize=(16, 6))
    plt.plot(epochs, train_scores, label="Train score")
    plt.plot(epochs, test_scores, label="Test score")
    plt.title("Train and test ROC AUC scores of the {}".format(model_name))
    plt.xlabel("Epoch")
    plt.ylabel("ROC AUC Score")
    plt.legend()
    plt.grid(True)
    plt.show()


def get_model_results(model_name, model_func):
    model = get_compiled_model(model_func)

    st = time.time()
    history = model.fit(
        train_dataset, epochs=5, validation_data=test_dataset, verbose=2
    )
    runtime = time.time() - st

    model.save("{}.h5".format(model_name))

    train_scores = history.history["auc"]
    test_scores = history.history["val_auc"]

    tf.keras.backend.clear_session()
    return (runtime, (train_scores, test_scores))




## === cell 11
runtime_base_maxpool, scores_base_maxpool = get_model_results(
    "model_base_maxpool", get_model_base_maxpool
)
plot_model_scores(scores_base_maxpool, "base + max pooling model")




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2341504248.py in <cell line: 0>()
      1 # Fix: previously training crashed before any model could be saved.
      2 # Keep original intent but ensure we at least train the model used for submission.
----> 3 runtime_base_maxpool, scores_base_maxpool = get_model_results(
      4     "model_base_maxpool", get_model_base_maxpool
      5 )

/tmp/ipykernel_11/2737109195.py in get_model_results(model_name, model_func)
     38     st = time.time()
     39     history = model.fit(
---> 40         train_dataset, epochs=5, validation_data=test_dataset, verbose=2
     41     )
     42     runtime = time.time() - st

NameError: name 'train_dataset' is not defined

## === cell 12
results = [
    ("Base + Max pooling", runtime_base_maxpool, scores_base_maxpool),
]

table = []
for i in range(len(results)):
    tmp = {
        "model": results[i][0],
        "runtime": results[i][1],
        "train_roc_auc_score": results[i][2][0][-1],
        "test_roc_auc_score": results[i][2][1][-1],
    }
    table.append(tmp)

leaderboard = pd.DataFrame(table).sort_values(
    by=["test_roc_auc_score", "runtime"], ascending=[False, True]
)
leaderboard



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2996044804.py in <cell line: 0>()
      1 # Keep leaderboard cell functional (now only one model is guaranteed to exist).
      2 results = [
----> 3     ("Base + Max pooling", runtime_base_maxpool, scores_base_maxpool),
      4 ]
      5 

NameError: name 'runtime_base_maxpool' is not defined

## === cell 13
model = load_model("model_base_maxpool.h5", compile=False)

submis_paths = (test_dir + sample_data["id"] + ".tif").values
submis_dataset = get_prefetched_data(submis_paths, BATCH_SIZE, cache=True)

pred = model.predict(submis_dataset, verbose=1).reshape(-1).astype(np.float32)
pred = np.clip(pred, 0.0, 1.0)

submission = pd.DataFrame({"id": sample_data["id"].values, "label": pred})
submission.to_csv("submission.csv", index=False)

submission.head(), submission.shape

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3397152890.py in <cell line: 0>()
      1 # Fix: submission previously failed because the model file was never created.
      2 # Now we load the trained maxpool model and generate predictions for test set.
----> 3 model = load_model("model_base_maxpool.h5", compile=False)
      4 
      5 submis_paths = (test_dir + sample_data["id"] + ".tif").values

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_api.py in load_model(filepath, custom_objects, compile, safe_mode)
    194         )
    195     if str(filepath).endswith((".h5", ".hdf5")):
--> 196         return legacy_h5_format.load_model_from_hdf5(
    197             filepath, custom_objects=custom_objects, compile=compile
    198         )

/usr/local/lib/python3.11/dist-packages/keras/src/legacy/saving/legacy_h5_format.py in load_model_from_hdf5(filepath, custom_objects, compile)
    114     opened_new_file = not isinstance(filepath, h5py.File)
    115     if opened_new_file:
--> 116         f = h5py.File(filepath, mode="r")
    117     else:
    118         f = filepath

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in __init__(self, name, mode, driver, libver, userblock_size, swmr, rdcc_nslots, rdcc_nbytes, rdcc_w0, track_order, fs_strategy, fs_persist, fs_threshold, fs_page_size, page_buf_size, min_meta_keep, min_raw_keep, locking, alignment_threshold, alignment_interval, meta_block_size, **kwds)
    562                                  fs_persist=fs_persist, fs_threshold=fs_threshold,
    563                                  fs_page_size=fs_page_size)
--> 564                 fid = make_fid(name, mode, userblock_size, fapl, fcpl, swmr=swmr)
    565 
    566             if isinstance(libver, tuple):

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in make_fid(name, mode, userblock_size, fapl, fcpl, swmr)
    236         if swmr and swmr_support:
    237             flags |= h5f.ACC_SWMR_READ
--> 238         fid = h5f.open(name, flags, fapl=fapl)
    239     elif mode == 'r+':
    240         fid = h5f.open(name, h5f.ACC_RDWR, fapl=fapl)

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/h5f.pyx in h5py.h5f.open()

FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = 'model_base_maxpool.h5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)
