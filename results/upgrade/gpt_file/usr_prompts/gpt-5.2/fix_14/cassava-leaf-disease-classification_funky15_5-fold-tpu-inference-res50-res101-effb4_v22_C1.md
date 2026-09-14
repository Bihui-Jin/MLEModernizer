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
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.8824418253248716

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

import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from PIL import Image

print("Tensorflow version " + tf.__version__)

tf.random.set_seed(42)
np.random.seed(42)

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass
try:
    tf.config.experimental.enable_tensor_float_32_execution(False)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
IMAGE_SIZE = 512
BATCH_SIZE = 64

NUM_CLASSES = 5

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
test_dir = os.path.join(DATA_ROOT, "test_images")
sample_sub_path = os.path.join(DATA_ROOT, "sample_submission.csv")

assert os.path.isdir(test_dir), f"test_dir not found: {test_dir}"
assert os.path.isfile(
    sample_sub_path
), f"sample_submission.csv not found: {sample_sub_path}"




## === cell 2
def build_model(image_size=512, num_classes=5):
    inputs = keras.Input(shape=(image_size, image_size, 3), name="image")
    base = keras.applications.EfficientNetB4(
        include_top=False, weights="imagenet", input_tensor=inputs, pooling="avg"
    )
    x = base.output
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="pred")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


modeleffb4_0 = build_model(IMAGE_SIZE, NUM_CLASSES)
mod_lst = [modeleffb4_0]

for _m in mod_lst:
    try:
        _m.compile(run_eagerly=False)
    except Exception:
        pass




## === cell 3
@tf.function
def _decode_resize_uint8_tf(path):
    b = tf.io.read_file(path)
    img = tf.image.decode_jpeg(b, channels=3)  # uint8, original size
    img_f = tf.image.resize(
        tf.cast(img, tf.float32),
        (IMAGE_SIZE, IMAGE_SIZE),
        method="bilinear",
        antialias=False,
    )
    img_u8 = tf.cast(tf.clip_by_value(tf.round(img_f), 0.0, 255.0), tf.uint8)
    return img_u8


@tf.function
def _tta_augment_stateless_tf(
    img_uint8, seed, crop_max=128, p_fliplr=0.5, p_flipud=0.5
):
    """
    img_uint8: HxWx3 uint8 tensor
    seed: int32 tensor shape [2] for stateless RNG
    returns: IMAGE_SIZExIMAGE_SIZEx3 uint8 tensor
    """
    img = img_uint8
    shape = tf.shape(img)
    h = shape[0]
    w = shape[1]

    seed0 = seed
    seed1 = tf.random.stateless_split(seed0, 2)[1]
    seed2 = tf.random.stateless_split(seed1, 2)[1]
    seed3 = tf.random.stateless_split(seed2, 2)[1]
    seed4 = tf.random.stateless_split(seed3, 2)[1]
    seed5 = tf.random.stateless_split(seed4, 2)[1]

    if crop_max > 0:
        h_lim = tf.minimum(tf.cast(crop_max, tf.int32), h // 4)
        w_lim = tf.minimum(tf.cast(crop_max, tf.int32), w // 4)

        top = tf.random.stateless_uniform([], seed0, 0, h_lim + 1, dtype=tf.int32)
        bottom = tf.random.stateless_uniform([], seed1, 0, h_lim + 1, dtype=tf.int32)
        left = tf.random.stateless_uniform([], seed2, 0, w_lim + 1, dtype=tf.int32)
        right = tf.random.stateless_uniform([], seed3, 0, w_lim + 1, dtype=tf.int32)

        end_h = tf.cond((h - bottom) > top, lambda: h - bottom, lambda: h)
        end_w = tf.cond((w - right) > left, lambda: w - right, lambda: w)
        img = img[top:end_h, left:end_w, :]

    img_f = tf.cast(img, tf.float32)
    img_f = tf.image.resize(
        img_f, (IMAGE_SIZE, IMAGE_SIZE), method="bilinear", antialias=False
    )
    img_u8 = tf.cast(tf.clip_by_value(tf.round(img_f), 0.0, 255.0), tf.uint8)

    r1 = tf.random.stateless_uniform([], seed4, 0.0, 1.0)
    img_u8 = tf.cond(
        r1 < p_fliplr, lambda: tf.image.flip_left_right(img_u8), lambda: img_u8
    )
    r2 = tf.random.stateless_uniform([], seed5, 0.0, 1.0)
    img_u8 = tf.cond(
        r2 < p_flipud, lambda: tf.image.flip_up_down(img_u8), lambda: img_u8
    )

    return img_u8


_load_image_uint8_tf = _decode_resize_uint8_tf




## === cell 4
def get_preds_model_list(
    image_dir, model_obj_list, TTA=True, aug_num=5, normalize=True
):
    image_paths = tf.io.gfile.glob(os.path.join(image_dir, "*.jpg"))
    image_paths = sorted(image_paths)
    if len(image_paths) == 0:
        raise FileNotFoundError(f"No .jpg files found in {image_dir}")

    img_ids = [os.path.basename(p) for p in image_paths]
    n = len(image_paths)

    options = tf.data.Options()
    options.experimental_deterministic = True

    paths_ds = tf.data.Dataset.from_tensor_slices(image_paths)

    @tf.function(reduce_retracing=True)
    def _infer_step(x):
        sum_over = None
        for m in model_obj_list:
            p = m(x, training=False)
            sum_over = p if sum_over is None else (sum_over + p)
        return sum_over / tf.cast(len(model_obj_list), tf.float32)

    if TTA:
        base_ds = paths_ds.map(
            _load_image_uint8_tf, num_parallel_calls=AUTOTUNE
        ).cache()

        idx_ds = tf.data.Dataset.range(n, output_type=tf.int32)
        aug_ds = tf.data.Dataset.range(aug_num, output_type=tf.int32)

        combo_ds = aug_ds.flat_map(lambda a: idx_ds.map(lambda i: (i, a)))

        seed_base = tf.constant([42, 12345], dtype=tf.int32)

        @tf.function(reduce_retracing=True)
        def _get_augmented(i, a):
            img_u8 = base_ds.skip(tf.cast(i, tf.int64)).take(1).get_single_element()
            seed = seed_base + tf.stack([i, a])
            img_u8 = _tta_augment_stateless_tf(img_u8, seed)
            x = tf.cast(img_u8, tf.float32)
            if normalize:
                x = x / 255.0
            return x, i  # keep original image index for aggregation

        ds = combo_ds.map(_get_augmented, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
        ds = ds.with_options(options)

        sum_logits = np.zeros((n, NUM_CLASSES), dtype=np.float32)

        for x, i in ds:
            p = _infer_step(x)  # [bs, C]
            p_np = p.numpy().astype(np.float32, copy=False)
            i_np = i.numpy().astype(np.int64, copy=False)
            np.add.at(sum_logits, i_np, p_np)

        avg_preds = sum_logits / float(aug_num)
        preds = np.argmax(avg_preds, axis=1).astype(np.int64)
        return pd.DataFrame({"image_id": img_ids, "label": preds.tolist()})

    else:
        ds = paths_ds.map(_load_image_uint8_tf, num_parallel_calls=AUTOTUNE)

        @tf.function(reduce_retracing=True)
        def _to_model_input(imgs_u8):
            x = tf.cast(imgs_u8, tf.float32)
            if normalize:
                x = x / 255.0
            return x

        ds = ds.batch(BATCH_SIZE, drop_remainder=False)
        ds = ds.map(_to_model_input, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
        ds = ds.with_options(options)

        preds_all = np.empty((n,), dtype=np.int64)
        out = 0
        for x in ds:
            p = _infer_step(x).numpy()
            bs = p.shape[0]
            preds_all[out : out + bs] = np.argmax(p, axis=1).astype(np.int64)
            out += bs

        return pd.DataFrame({"image_id": img_ids, "label": preds_all.tolist()})


predict_df = get_preds_model_list(
    test_dir, mod_lst, normalize=True, aug_num=4, TTA=True
)

sample_sub = pd.read_csv(sample_sub_path)
submission = sample_sub[["image_id"]].merge(predict_df, on="image_id", how="left")
submission["label"] = submission["label"].fillna(0).astype(int)

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3973497328.py in <cell line: 0>()
     93 
     94 
---> 95 predict_df = get_preds_model_list(
     96     test_dir, mod_lst, normalize=True, aug_num=4, TTA=True
     97 )

/tmp/ipykernel_11/3973497328.py in get_preds_model_list(image_dir, model_obj_list, TTA, aug_num, normalize)
     52 
     53         # Speed: vectorized pipeline with parallel map + prefetch.
---> 54         ds = combo_ds.map(_get_augmented, num_parallel_calls=AUTOTUNE)
     55         ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
     56         ds = ds.with_options(options)

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

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/tmp/__autograph_generated_filedtdg692h.py in tf___get_augmented(i, a)
     13                 img_u8 = ag__.converted_call(ag__.converted_call(ag__.converted_call(ag__.ld(base_ds).skip, (ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(i), ag__.ld(tf).int64), None, fscope),), None, fscope).take, (1,), None, fscope).get_single_element, (), None, fscope)
     14                 seed = ag__.ld(seed_base) + ag__.converted_call(ag__.ld(tf).stack, ([ag__.ld(i), ag__.ld(a)],), None, fscope)
---> 15                 img_u8 = ag__.converted_call(ag__.ld(_tta_augment_stateless_tf), (ag__.ld(img_u8), ag__.ld(seed)), None, fscope)
     16                 x = ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(img_u8), ag__.ld(tf).float32), None, fscope)
     17 

/tmp/__autograph_generated_file8owkk5zh.py in tf___tta_augment_stateless_tf(img_uint8, seed, crop_max, p_fliplr, p_flipud)
     18                 w = ag__.ld(shape)[1]
     19                 seed0 = ag__.ld(seed)
---> 20                 seed1 = ag__.converted_call(ag__.ld(tf).random.stateless_split, (ag__.ld(seed0), 2), None, fscope)[1]
     21                 seed2 = ag__.converted_call(ag__.ld(tf).random.stateless_split, (ag__.ld(seed1), 2), None, fscope)[1]
     22                 seed3 = ag__.converted_call(ag__.ld(tf).random.stateless_split, (ag__.ld(seed2), 2), None, fscope)[1]

AttributeError: in user code:

    File "/tmp/ipykernel_11/3973497328.py", line 47, in _get_augmented  *
        img_u8 = _tta_augment_stateless_tf(img_u8, seed)
    File "/tmp/ipykernel_11/1622514021.py", line 34, in _tta_augment_stateless_tf  *
        seed1 = tf.random.stateless_split(seed0, 2)[1]

    AttributeError: module 'tensorflow._api.v2.random' has no attribute 'stateless_split'


## === cell 5
try:
    from IPython.display import display

    display(submission.head(10))
except Exception as e:
    print("Display not available:", e)
