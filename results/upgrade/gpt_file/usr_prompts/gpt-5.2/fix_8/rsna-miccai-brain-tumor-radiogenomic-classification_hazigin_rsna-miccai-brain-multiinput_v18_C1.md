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
Predict the genetic subtype of glioblastoma using MRI (magnetic resonance imaging) scans to detect for the presence of MGMT promoter methylation.

## Metric
Area under the ROC curve between the predicted probability and the observed target.

## Submission Format
For each `BraTS21ID` in the test set, you must predict a probability for the target `MGMT_value`. The file should contain a header and have the following format:

```
BraTS21ID,MGMT_value
00001,0.5
00013,0.5
00015,0.5
etc.
```

## Dataset
- **train/** - folder containing the training files, with each top-level folder representing a subject. **NOTE:** There are some unexpected issues with the following three cases in the training dataset, participants can exclude the cases during training: `[00109, 00123, 00709]`. We have checked and confirmed that the testing dataset is free from such issues.
- **train_labels.csv** - file containing the target `MGMT_value` for each subject in the training data (e.g. the presence of MGMT promoter methylation)
- **test/** - the test files, which use the same structure as `train/`; your task is to predict the `MGMT_value` for each subject in the test data. **NOTE**: the total size of the rerun test set (Public and Private) is ~5x the size of the Public test set
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (202 lines)
            sample_submission.csv (60 lines)
            sample_submission.csv.zip (382 Bytes)
            test.zip (1.3 GB)
            train.zip (10.2 GB)
            train_labels.csv (527 lines)
            train_labels.csv.zip (1.4 kB)
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
            test/
                00002/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 29 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 382 other files
                00019/
                    FLAIR/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 30 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-258.dcm (525.4 kB)
                        Image-259.dcm (525.4 kB)
                        ... and 127 other files
                ... and 58 other folders
            train/
                00000/
                    FLAIR/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 398 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 406 other files
                00003/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 406 other files
                ... and 525 other folders
        input/
            description.md (202 lines)
            sample_submission.csv (60 lines)
            sample_submission.csv.zip (382 Bytes)
            test.zip (1.3 GB)
            train.zip (10.2 GB)
            train_labels.csv (527 lines)
            train_labels.csv.zip (1.4 kB)
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
            test/
                00002/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 29 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 382 other files
                00019/
                    FLAIR/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 30 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-258.dcm (525.4 kB)
                        Image-259.dcm (525.4 kB)
                        ... and 127 other files
                ... and 58 other folders
            train/
                00000/
                    FLAIR/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 398 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 406 other files
                00003/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 406 other files
                ... and 525 other folders
        working/
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
```

-> data/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> (stopped after 10 files for performance)

# 5. Target score

-1.0

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import random
import cv2
import numpy as np
import gc
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from pathlib import Path
import math

import tensorflow_io as tfio

try:
    tf.config.threading.set_intra_op_parallelism_threads(
        max(1, (os.cpu_count() or 4) // 2)
    )
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

print("TF:", tf.__version__)
print("TFIO:", tfio.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
df_preds = pd.read_csv(
    "../input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv"
)
weightdatapath = Path("../input/weight-multi-20210919")
testdatapaht = "../input/rsna-miccai-brain-tumor-radiogenomic-classification/test"



## === cell 2
height = 256
width = 256
channel = 3
batch_size = 32
epochs = 400
seed = 26
epoch = "0921"
views = ["FLAIR", "T1w", "T1wCE", "T2w"]




## === cell 3
def set_seed(seed=200):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(seed)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass




## === cell 4
@tf.keras.utils.register_keras_serializable(package="Custom")
def mish(inputs):
    return inputs * tf.math.tanh(tf.math.softplus(inputs))


@tf.keras.utils.register_keras_serializable(package="Custom")
def siren(inputs):
    return 1.0 / (1.0 + tf.math.exp(-inputs))


tf.keras.utils.get_custom_objects()["mish"] = mish
tf.keras.utils.get_custom_objects()["siren"] = siren




## === cell 5
class Silu(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(Silu, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name="kernel",
            shape=[int(input_shape[-1]), self.num_outputs],
            initializer="glorot_uniform",
            trainable=True,
        )

    def call(self, input):
        return tf.nn.silu(input)




## === cell 6

_DCM_LIST_CACHE = {}


def _list_dcm_files(idx: str, view: str):
    key = (idx, view)
    if key in _DCM_LIST_CACHE:
        return _DCM_LIST_CACHE[key]

    base_dir = os.path.join(testdatapaht, idx, view)
    if not os.path.isdir(base_dir):
        files = []
    else:
        files = []
        for root, _, names in os.walk(base_dir):
            for name in names:
                if name.lower().endswith(".dcm"):
                    files.append(os.path.join(root, name))
        files.sort()
    _DCM_LIST_CACHE[key] = files
    return files


def _decode_resize_dicom(path):
    dcm_bytes = tf.io.read_file(path)
    img = tfio.image.decode_dicom_image(
        dcm_bytes,
        dtype=tf.uint16,
        color_dim=False,
        on_error="skip",
        scale="preserve",
    )
    img = tf.cast(img, tf.float32)
    img = tf.squeeze(img)  # [H,W] or [frames,H,W]
    img = tf.cond(tf.equal(tf.rank(img), 3), lambda: img[0], lambda: img)
    img = tf.expand_dims(img, axis=-1)  # [H,W,1]
    img = tf.image.resize(
        img, (height, width), method=tf.image.ResizeMethod.AREA, antialias=False
    )
    img = tf.squeeze(img, axis=-1)  # [height,width]
    return img


def load_imgs(idx, view, ignore_zeros=True):
    files = _list_dcm_files(idx, view)
    if len(files) == 0:
        return np.zeros((1, height, width), dtype=np.float32)

    ds = tf.data.Dataset.from_tensor_slices(files)
    ds = ds.map(
        _decode_resize_dicom, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
    )
    imgs = tf.stack(list(ds.as_numpy_iterator()), axis=0).astype(np.float32, copy=False)
    if imgs.shape[0] == 0:
        imgs = np.zeros((1, height, width), dtype=np.float32)
    return imgs




## === cell 7
dim = (height, width)
t_imgs = np.empty((channel, *dim), dtype=np.float32)


def data_generation(ID, view, is_Train=True):
    idx = str(ID).zfill(5)
    imgs = load_imgs(idx, view, ignore_zeros=False)  # already float32
    t_size0 = imgs.shape[0]
    t_a = math.ceil(t_size0 / 3)

    t_img = imgs[:t_a]
    t_imgs[0] = t_img.mean(axis=0) * 0.3

    t_img = imgs[t_a : t_size0 - t_a]
    if t_img.shape[0] == 0:
        t_img = imgs
    t_imgs[1] = t_img.mean(axis=0) * 0.4

    t_img = imgs[t_size0 - t_a :]
    t_imgs[2] = t_img.mean(axis=0) * 0.3

    img_ = t_imgs.transpose(1, 2, 0)
    return img_




## === cell 8
def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))




## === cell 9
def serialize_example_test(feature0):
    feature0 = np.asarray(feature0, dtype=np.float32, order="C")
    feature = {"image": _bytes_feature(feature0.tobytes())}
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()




## === cell 10
def _build_one_view_tfrec(view_name, ids, out_path, num_workers=None, chunksize=64):
    def _worker(id_):
        img = data_generation(id_, view_name, False)
        return serialize_example_test(img)

    with tf.io.TFRecordWriter(out_path) as writer:
        buf = []
        for x in tqdm(ids, desc=f"TFREC {view_name}", leave=False):
            buf.append(_worker(x))
            if len(buf) >= chunksize:
                for rec in buf:
                    writer.write(rec)
                buf.clear()
        if buf:
            for rec in buf:
                writer.write(rec)
            buf.clear()




## === cell 11
ids = df_preds["BraTS21ID"].tolist()
for v in views:
    out_path = f"./brain_test_{v}.tfrec"
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        _build_one_view_tfrec(v, ids, out_path, num_workers=1, chunksize=64)




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NotImplementedError                       Traceback (most recent call last)
/tmp/ipykernel_11/2907540507.py in <cell line: 0>()
      3     out_path = f"./brain_test_{v}.tfrec"
      4     if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
----> 5         _build_one_view_tfrec(v, ids, out_path, num_workers=1, chunksize=64)
      6 
      7 

/tmp/ipykernel_11/3031140183.py in _build_one_view_tfrec(view_name, ids, out_path, num_workers, chunksize)
      8         buf = []
      9         for x in tqdm(ids, desc=f"TFREC {view_name}", leave=False):
---> 10             buf.append(_worker(x))
     11             if len(buf) >= chunksize:
     12                 for rec in buf:

/tmp/ipykernel_11/3031140183.py in _worker(id_)
      2 def _build_one_view_tfrec(view_name, ids, out_path, num_workers=None, chunksize=64):
      3     def _worker(id_):
----> 4         img = data_generation(id_, view_name, False)
      5         return serialize_example_test(img)
      6 

/tmp/ipykernel_11/2495708461.py in data_generation(ID, view, is_Train)
      6 def data_generation(ID, view, is_Train=True):
      7     idx = str(ID).zfill(5)
----> 8     imgs = load_imgs(idx, view, ignore_zeros=False)  # already float32
      9     t_size0 = imgs.shape[0]
     10     t_a = math.ceil(t_size0 / 3)

/tmp/ipykernel_11/2076983041.py in load_imgs(idx, view, ignore_zeros)
     59     ds = tf.data.Dataset.from_tensor_slices(files)
     60     # Deterministic order preserved because input is already sorted and determinism is enabled.
---> 61     ds = ds.map(
     62         _decode_resize_dicom, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
     63     )

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

/tmp/__autograph_generated_fileg_rc6h3i.py in tf___decode_resize_dicom(path)
      9                 retval_ = ag__.UndefinedReturnValue()
     10                 dcm_bytes = ag__.converted_call(ag__.ld(tf).io.read_file, (ag__.ld(path),), None, fscope)
---> 11                 img = ag__.converted_call(ag__.ld(tfio).image.decode_dicom_image, (ag__.ld(dcm_bytes),), dict(dtype=ag__.ld(tf).uint16, color_dim=False, on_error='skip', scale='preserve'), fscope)
     12                 img = ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(img), ag__.ld(tf).float32), None, fscope)
     13                 img = ag__.converted_call(ag__.ld(tf).squeeze, (ag__.ld(img),), None, fscope)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    432     if is_autograph_strict_conversion_mode():
    433       raise
--> 434     return _fall_back_unconverted(f, args, kwargs, options, e)
    435 
    436   with StackTraceMapper(converted_f), tf_stack.CurrentModuleFilter():

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in _fall_back_unconverted(f, args, kwargs, options, exc)
    483     logging.warning(warning_template, f, file_bug_message, exc)
    484 
--> 485   return _call_unconverted(f, args, kwargs, options)
    486 
    487 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in _call_unconverted(f, args, kwargs, options, update_cache)
    457 
    458   if kwargs is not None:
--> 459     return f(*args, **kwargs)
    460   return f(*args)
    461 

/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/dicom_ops.py in decode_dicom_image(contents, color_dim, on_error, scale, dtype, name)
     82         A `Tensor` of type `dtype` and the shape is determined by the DICOM file.
     83     """
---> 84     return core_ops.io_decode_dicom_image(
     85         contents=contents,
     86         color_dim=color_dim,

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

    File "/tmp/ipykernel_11/2076983041.py", line 34, in _decode_resize_dicom  *
        img = tfio.image.decode_dicom_image(
    File "/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/dicom_ops.py", line 84, in decode_dicom_image  **
        return core_ops.io_decode_dicom_image(
    File "/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/__init__.py", line 88, in __getattr__
        return getattr(self._load(), attrb)
    File "/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/__init__.py", line 84, in _load
        self._mod = _load_library(self._library)
    File "/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/__init__.py", line 69, in _load_library
        raise NotImplementedError(

    NotImplementedError: unable to open file: libtensorflow_io.so, from paths: ['/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/libtensorflow_io.so']
    caused by: ['/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/libtensorflow_io.so: undefined symbol: _ZN3tsl8str_util9LowercaseB5cxx11ESt17basic_string_viewIcSt11char_traitsIcEE']


## === cell 12
def deserialize_example(serialized_string):
    image_feature_description = {"image": tf.io.FixedLenFeature([], tf.string)}
    parsed_record = tf.io.parse_single_example(
        serialized_string, image_feature_description
    )
    image = tf.io.decode_raw(parsed_record["image"], tf.float32)
    image = tf.reshape(image, [height, width, channel])
    return image




## === cell 13
def argument_image_tw2_val(img):
    img = tf.cast(img, tf.float32) / 255.0
    return img




## === cell 14
class RegressionModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super(RegressionModel, self).__init__(**kwargs)
        self.conv1 = tf.keras.layers.Conv2D(
            128, 5, strides=2, kernel_initializer="he_normal", activation="mish"
        )
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.rule1 = Silu(0)
        self.rule1 = tf.keras.layers.LeakyReLU(alpha=0.3)
        self.max1 = tf.keras.layers.MaxPooling2D(5)

        self.conv2 = tf.keras.layers.Conv2D(
            128, 5, strides=2, kernel_initializer="he_normal", activation="mish"
        )
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.rule2 = Silu(0)
        self.max2 = tf.keras.layers.MaxPooling2D(5)

        self.conv3 = tf.keras.layers.Conv2D(
            128, 5, strides=2, kernel_initializer="he_normal", activation="mish"
        )
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.rule3 = Silu(0)
        self.max3 = tf.keras.layers.MaxPooling2D(5)

        self.conv4 = tf.keras.layers.Conv2D(
            128, 5, strides=2, kernel_initializer="he_normal", activation="mish"
        )
        self.bn4 = tf.keras.layers.BatchNormalization()
        self.rule4 = Silu(0)
        self.max4 = tf.keras.layers.MaxPooling2D(5)

        para_relu = tf.keras.layers.LeakyReLU(alpha=0.5)
        self.dence256 = tf.keras.layers.Dense(
            256, kernel_initializer="he_normal", activation="mish"
        )
        self.dence128 = tf.keras.layers.Dense(
            128, kernel_initializer="he_normal", activation="swish"
        )
        self.dence64 = tf.keras.layers.Dense(
            64, kernel_initializer="he_normal", activation="elu"
        )
        self.dence256_2 = tf.keras.layers.Dense(
            256, kernel_initializer="he_normal", activation="mish"
        )
        self.dence128_2 = tf.keras.layers.Dense(
            128, kernel_initializer="he_normal", activation="swish"
        )
        self.dence64_2 = tf.keras.layers.Dense(
            64, kernel_initializer="he_normal", activation="elu"
        )
        self.dence256_3 = tf.keras.layers.Dense(
            256, kernel_initializer="he_normal", activation="mish"
        )
        self.dence128_3 = tf.keras.layers.Dense(
            128, kernel_initializer="he_normal", activation="swish"
        )
        self.dence64_3 = tf.keras.layers.Dense(
            64, kernel_initializer="he_normal", activation="elu"
        )
        self.dence256_4 = tf.keras.layers.Dense(
            256, kernel_initializer="he_normal", activation="mish"
        )
        self.dence128_4 = tf.keras.layers.Dense(
            128, kernel_initializer="he_normal", activation="swish"
        )
        self.dence64_4 = tf.keras.layers.Dense(
            64, kernel_initializer="he_normal", activation="elu"
        )

        self.dence32 = tf.keras.layers.Dense(
            32, kernel_initializer="he_normal", activation="siren"
        )
        self.dence256_5 = tf.keras.layers.Dense(
            256, kernel_initializer="he_normal", activation="mish"
        )
        self.dence128_5 = tf.keras.layers.Dense(
            128, kernel_initializer="he_normal", activation="swish"
        )
        self.dence64_5 = tf.keras.layers.Dense(
            64, kernel_initializer="he_normal", activation=para_relu
        )
        self.dropoup5 = tf.keras.layers.Dropout(0.2)
        self.dropoup4 = tf.keras.layers.Dropout(0.5)
        self.dropoup4_2 = tf.keras.layers.Dropout(0.5)
        self.dropoup4_3 = tf.keras.layers.Dropout(0.5)
        self.dropoup4_4 = tf.keras.layers.Dropout(0.5)
        self.dropoup3 = tf.keras.layers.Dropout(0.2)
        self.dropoup3_2 = tf.keras.layers.Dropout(0.2)
        self.dropoup3_3 = tf.keras.layers.Dropout(0.2)
        self.dropoup3_4 = tf.keras.layers.Dropout(0.2)
        self.dropoup2 = tf.keras.layers.Dropout(0.2)
        self.dropoup2_2 = tf.keras.layers.Dropout(0.2)
        self.dropoup2_3 = tf.keras.layers.Dropout(0.2)
        self.dropoup2_4 = tf.keras.layers.Dropout(0.2)
        self.dropoup = tf.keras.layers.Dropout(0.1)
        self.dence1 = tf.keras.layers.Dense(1, activation="sigmoid")

        self.flatten = tf.keras.layers.Flatten()

    def call(self, input_tensor, training=True):
        x1 = input_tensor[0]
        x1 = self.conv1(x1)
        x1 = self.bn1(x1, training=training)
        x1 = self.rule1(x1)
        x1 = self.max1(x1)
        x1 = self.dence256(x1)
        x1 = self.dropoup4(x1, training=training)
        x1 = self.dence128(x1)

        x2 = self.conv2(input_tensor[1])
        x2 = self.bn2(x2, training=training)
        x2 = self.rule2(x2)
        x2 = self.max2(x2)
        x2 = self.dence256_2(x2)
        x2 = self.dropoup4_2(x2, training=training)
        x2 = self.dence128_2(x2)

        x3 = self.conv3(input_tensor[2])
        x3 = self.bn3(x3, training=training)
        x3 = self.rule3(x3)
        x3 = self.max3(x3)
        x3 = self.dence256_3(x3)
        x3 = self.dropoup4_3(x3, training=training)
        x3 = self.dence128_3(x3)

        x4 = self.conv4(input_tensor[3])
        x4 = self.bn4(x4, training=training)
        x4 = self.rule4(x4)
        x4 = self.max4(x4)
        x4 = self.dence256_4(x4)
        x4 = self.dropoup4_4(x4, training=training)
        x4 = self.dence128_4(x4)

        x = tf.keras.layers.Concatenate()([x1, x2, x3, x4])
        x = self.dence256_5(x)
        x = self.dropoup3_4(x, training=training)
        x = self.dence128_5(x)
        x = self.dropoup5(x, training=training)
        x = self.flatten(x)
        x = self.dence64_5(x)
        x = self.dropoup(x, training=training)
        x = self.dence32(x)

        return self.dence1(x)

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            predictions = self(x, training=True)
            loss = self.compiled_loss(y, predictions, regularization_losses=self.losses)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(y, predictions)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}




## === cell 15
loss_func = tf.keras.losses.BinaryCrossentropy(from_logits=False)
opt = tf.keras.optimizers.SGD(
    learning_rate=1e-5, decay=1e-6, momentum=0.9, nesterov=True
)
f_pre = []

AUTOTUNE = tf.data.AUTOTUNE
options = tf.data.Options()
options.experimental_deterministic = True


def make_testset(path):
    ds = tf.data.TFRecordDataset(path, num_parallel_reads=AUTOTUNE)
    ds = ds.with_options(options)
    ds = ds.map(deserialize_example, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.map(argument_image_tw2_val, num_parallel_calls=AUTOTUNE, deterministic=True)
    return ds


testset0 = make_testset("./brain_test_FLAIR.tfrec")
testset1 = make_testset("./brain_test_T1w.tfrec")
testset2 = make_testset("./brain_test_T1wCE.tfrec")
testset3 = make_testset("./brain_test_T2w.tfrec")

input_a = tf.keras.Input(shape=(height, width, channel), name="input_a")
input_b = tf.keras.Input(shape=(height, width, channel), name="input_b")
input_c = tf.keras.Input(shape=(height, width, channel), name="input_c")
input_d = tf.keras.Input(shape=(height, width, channel), name="input_d")

model = RegressionModel()
model([input_a, input_b, input_c, input_d])

model.compile(
    optimizer=opt,
    loss=loss_func,
    metrics=[
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.BinaryCrossentropy(name="bce"),
    ],
)

test_ds = tf.data.Dataset.zip((testset0, testset1, testset2, testset3))
test_ds = test_ds.batch(10, drop_remainder=False).prefetch(AUTOTUNE)

loaded_any = False
for f in range(5):  # Fold
    t_weight = "weight-Regression-multi_fold_0" + str(f) + "-" + epoch + ".ckpt"
    w_path = Path(weightdatapath, t_weight)
    try:
        model.load_weights(w_path)
        loaded_any = True
    except Exception as e:
        print("WARNING: could not load weights:", str(w_path), "|", repr(e))
        continue

    preds = model.predict(test_ds, batch_size=batch_size, verbose=0)
    f_pre.append(preds)

if loaded_any and len(f_pre) > 0:
    pre = np.array(f_pre)  # (folds, N, 1)
    finpre = pre.mean(axis=0)
    finpre = np.asarray(finpre).reshape(-1)  # (N,)
else:
    finpre = np.full((len(df_preds),), 0.5, dtype=np.float32)

if finpre.shape[0] != len(df_preds):
    print(
        "WARNING: prediction length mismatch:",
        finpre.shape[0],
        "vs",
        len(df_preds),
        "-> using 0.5 fallback.",
    )
    finpre = np.full((len(df_preds),), 0.5, dtype=np.float32)

df_preds["MGMT_value"] = finpre.astype(np.float32)

subfilename = "submission.csv"
df_preds.to_csv(subfilename, index=False)
print("Wrote:", subfilename, "rows:", len(df_preds), "cols:", list(df_preds.columns))
print(df_preds.head())

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/66881356.py in <cell line: 0>()
     28 input_d = tf.keras.Input(shape=(height, width, channel), name="input_d")
     29 
---> 30 model = RegressionModel()
     31 model([input_a, input_b, input_c, input_d])
     32 

/tmp/ipykernel_11/3141648079.py in __init__(self, **kwargs)
     69         )
     70 
---> 71         self.dence32 = tf.keras.layers.Dense(
     72             32, kernel_initializer="he_normal", activation="siren"
     73         )

/usr/local/lib/python3.11/dist-packages/keras/src/layers/core/dense.py in __init__(self, units, activation, use_bias, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint, lora_rank, **kwargs)
     87         super().__init__(activity_regularizer=activity_regularizer, **kwargs)
     88         self.units = units
---> 89         self.activation = activations.get(activation)
     90         self.use_bias = use_bias
     91         self.kernel_initializer = initializers.get(kernel_initializer)

/usr/local/lib/python3.11/dist-packages/keras/src/activations/__init__.py in get(identifier)
    124     if callable(obj):
    125         return obj
--> 126     raise ValueError(
    127         f"Could not interpret activation function identifier: {identifier}"
    128     )

ValueError: Could not interpret activation function identifier: siren
