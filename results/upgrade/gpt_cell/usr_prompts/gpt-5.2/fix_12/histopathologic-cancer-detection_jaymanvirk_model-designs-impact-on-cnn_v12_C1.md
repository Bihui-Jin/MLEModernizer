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

3.12

# 2. Installed packages

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

# 3. Data file paths

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

# 4. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

try:
    from google.protobuf import message_factory as _message_factory

    if not hasattr(_message_factory.MessageFactory, "GetPrototype"):

        if hasattr(_message_factory.MessageFactory, "GetMessageClass"):

            def _GetPrototype(self, descriptor):
                return self.GetMessageClass(descriptor)

        else:

            def _GetPrototype(self, descriptor):
                return getattr(descriptor, "_concrete_class", None)

        _message_factory.MessageFactory.GetPrototype = _GetPrototype
except Exception:
    pass

import time
import random
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import tensorflow as tf

try:
    import tensorflow_io as tfio
except Exception:
    tfio = None

from sklearn.utils import resample
from sklearn.model_selection import train_test_split

from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass



## === cell 1
input_dir = "/kaggle/input/histopathologic-cancer-detection"
list_l = [os.path.join(input_dir, x) for x in os.listdir(input_dir)]
list_l



## === cell 2
sample_csv_path = os.path.join(input_dir, "sample_submission.csv")
train_csv_path = os.path.join(input_dir, "train_labels.csv")
train_dir_path = os.path.join(input_dir, "train")
test_dir_path = os.path.join(input_dir, "test")

sample_data = pd.read_csv(sample_csv_path)
train_data = pd.read_csv(train_csv_path)

train_dir = train_dir_path + "/"
test_dir = test_dir_path + "/"

list_l = [sample_csv_path, train_csv_path, test_dir_path, train_dir_path]



## === cell 3
del list_l




## === cell 4
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




## === cell 5
print_short_summary("Train data", train_data)



## === cell 6
print_short_summary("Sample data", sample_data)



## === cell 7
print_number_files(train_dir)



## === cell 8
print_number_files(test_dir)



## === cell 9
del print_short_summary, print_number_files



## === cell 10
pass



## === cell 11
pass



## === cell 12
pass



## === cell 13
pass



## === cell 14
pass



## === cell 15
pass



## === cell 16
no_cancer = train_data[train_data["label"] == 0]
cancer = train_data[train_data["label"] == 1]

no_cancer_downsampled = resample(
    no_cancer, replace=False, n_samples=len(cancer), random_state=0
)

balanced_train_data = pd.concat([no_cancer_downsampled, cancer])
balanced_train_data = balanced_train_data.sample(frac=1, random_state=0).reset_index(
    drop=True
)



## === cell 17
del no_cancer, cancer, no_cancer_downsampled



## === cell 18
image_paths = train_dir + balanced_train_data["id"] + ".tif"
image_paths = image_paths.values

labels = balanced_train_data["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    image_paths, labels, test_size=0.25, shuffle=True, random_state=0
)



## === cell 19
del image_paths, labels




## === cell 20
def get_decoded_image(image_path, label=None):
    """
    Load and preprocess images using TensorFlow I/O.
    Decode image with 4 channels RGBA.
    Resize image to 32x32px.
    Scale pixels from 0 to 1.
    Args:
        image_path: path to TIFF image
        label (optional): true label from train data
    Returns:
        (img, label): for train data
        img: for test data
    """
    img = tf.io.read_file(image_path)
    img = tfio.experimental.image.decode_tiff(img)
    img = tf.image.resize(img, [32, 32])
    img = tf.cast(img, tf.float32) / 255.0

    return img if label is None else (img, label)


def get_prefetched_data(data, batch_size, cache=False):
    """
    Create a TensorFlow dataset from image paths and labels.
    Execution in parallel.
    Load, preprocess images and batch the data.
    Prefetch batches to improve training performance.
    Args:
        data (tuple): image paths and corresponding labels OR image paths only
        batch_size (int): number of samples per batch
        cache (bool): cache preprocessed samples in memory for reuse across epochs
    Returns:
        tf.data.Dataset: preprocessed and preloaded TensorFlow dataset for keras CNN
    """
    AUTOTUNE = tf.data.AUTOTUNE

    dataset = tf.data.Dataset.from_tensor_slices(data)

    options = tf.data.Options()
    options.experimental_deterministic = True
    dataset = dataset.with_options(options)

    dataset = dataset.map(get_decoded_image, num_parallel_calls=AUTOTUNE)

    if cache:
        dataset = dataset.cache()

    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    return dataset




## === cell 21
if tfio is None:
    _use_tfio = False
else:
    try:
        _use_tfio = hasattr(tfio.experimental.image, "decode_tiff")
    except Exception:
        _use_tfio = False

if _use_tfio:

    def get_decoded_image(image_path, label=None):
        img = tf.io.read_file(image_path)
        img = tfio.experimental.image.decode_tiff(
            img
        )  # typically returns HxWx3 or HxWx4
        img = img[..., :3]
        img = tf.image.resize(img, [32, 32])
        img = tf.cast(img, tf.float32) / 255.0
        alpha = tf.ones([32, 32, 1], dtype=img.dtype)
        img = tf.concat([img, alpha], axis=-1)
        return img if label is None else (img, label)

else:

    def get_decoded_image(image_path, label=None):
        img_bytes = tf.io.read_file(image_path)

        def _decode_with_pil_rgb(x):
            import io
            import numpy as _np
            from PIL import Image as _Image

            im = _Image.open(io.BytesIO(x))
            im = im.convert("RGB")
            return _np.array(im, dtype=_np.uint8)

        img = tf.numpy_function(_decode_with_pil_rgb, [img_bytes], tf.uint8)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, [32, 32])
        img = tf.cast(img, tf.float32) / 255.0
        alpha = tf.ones([32, 32, 1], dtype=img.dtype)
        img = tf.concat([img, alpha], axis=-1)
        return img if label is None else (img, label)


BATCH_SIZE = 128

train_dataset = get_prefetched_data((X_train, y_train), BATCH_SIZE, cache=True)
test_dataset = get_prefetched_data((X_test, y_test), BATCH_SIZE, cache=True)



## --- ERROR in cell 21, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mNotImplementedError[0m                       Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3550011738.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     51[0m [0;34m[0m[0m
[1;32m     52[0m [0;31m# --- Speed: cache train/val datasets to avoid repeated TIFF decode across epochs.[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 53[0;31m [0mtrain_dataset[0m [0;34m=[0m [0mget_prefetched_data[0m[0;34m([0m[0;34m([0m[0mX_train[0m[0;34m,[0m [0my_train[0m[0;34m)[0m[0;34m,[0m [0mBATCH_SIZE[0m[0;34m,[0m [0mcache[0m[0;34m=[0m[0;32mTrue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     54[0m [0mtest_dataset[0m [0;34m=[0m [0mget_prefetched_data[0m[0;34m([0m[0;34m([0m[0mX_test[0m[0;34m,[0m [0my_test[0m[0;34m)[0m[0;34m,[0m [0mBATCH_SIZE[0m[0;34m,[0m [0mcache[0m[0;34m=[0m[0;32mTrue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     55[0m [0;34m[0m[0m

[0;32m/tmp/ipykernel_11/940018422.py[0m in [0;36mget_prefetched_data[0;34m(data, batch_size, cache)[0m
[1;32m     42[0m     [0mdataset[0m [0;34m=[0m [0mdataset[0m[0;34m.[0m[0mwith_options[0m[0;34m([0m[0moptions[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     43[0m [0;34m[0m[0m
[0;32m---> 44[0;31m     [0mdataset[0m [0;34m=[0m [0mdataset[0m[0;34m.[0m[0mmap[0m[0;34m([0m[0mget_decoded_image[0m[0;34m,[0m [0mnum_parallel_calls[0m[0;34m=[0m[0mAUTOTUNE[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     45[0m [0;34m[0m[0m
[1;32m     46[0m     [0;31m# --- Speed: caching avoids re-decoding TIFFs every epoch; exact values reused, so correctness preserved.[0m[0;34m[0m[0;34m[0m[0m

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

[0;32m/tmp/__autograph_generated_filenwks84ft.py[0m in [0;36mtf__get_decoded_image[0;34m(image_path, label)[0m
[1;32m      9[0m                 [0mretval_[0m [0;34m=[0m [0mag__[0m[0;34m.[0m[0mUndefinedReturnValue[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     10[0m                 [0mimg[0m [0;34m=[0m [0mag__[0m[0;34m.[0m[0mconverted_call[0m[0;34m([0m[0mag__[0m[0;34m.[0m[0mld[0m[0;34m([0m[0mtf[0m[0;34m)[0m[0;34m.[0m[0mio[0m[0;34m.[0m[0mread_file[0m[0;34m,[0m [0;34m([0m[0mag__[0m[0;34m.[0m[0mld[0m[0;34m([0m[0mimage_path[0m[0;34m)[0m[0;34m,[0m[0;34m)[0m[0;34m,[0m [0;32mNone[0m[0;34m,[0m [0mfscope[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 11[0;31m                 [0mimg[0m [0;34m=[0m [0mag__[0m[0;34m.[0m[0mconverted_call[0m[0;34m([0m[0mag__[0m[0;34m.[0m[0mld[0m[0;34m([0m[0mtfio[0m[0;34m)[0m[0;34m.[0m[0mexperimental[0m[0;34m.[0m[0mimage[0m[0;34m.[0m[0mdecode_tiff[0m[0;34m,[0m [0;34m([0m[0mag__[0m[0;34m.[0m[0mld[0m[0;34m([0m[0mimg[0m[0;34m)[0m[0;34m,[0m[0;34m)[0m[0;34m,[0m [0;32mNone[0m[0;34m,[0m [0mfscope[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     12[0m                 [0mimg[0m [0;34m=[0m [0mag__[0m[0;34m.[0m[0mld[0m[0;34m([0m[0mimg[0m[0;34m)[0m[0;34m[[0m[0;34m...[0m[0;34m,[0m [0;34m:[0m[0;36m3[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[1;32m     13[0m                 [0mimg[0m [0;34m=[0m [0mag__[0m[0;34m.[0m[0mconverted_call[0m[0;34m([0m[0mag__[0m[0;34m.[0m[0mld[0m[0;34m([0m[0mtf[0m[0;34m)[0m[0;34m.[0m[0mimage[0m[0;34m.[0m[0mresize[0m[0;34m,[0m [0;34m([0m[0mag__[0m[0;34m.[0m[0mld[0m[0;34m([0m[0mimg[0m[0;34m)[0m[0;34m,[0m [0;34m[[0m[0;36m32[0m[0;34m,[0m [0;36m32[0m[0;34m][0m[0;34m)[0m[0;34m,[0m [0;32mNone[0m[0;34m,[0m [0mfscope[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py[0m in [0;36mconverted_call[0;34m(f, args, kwargs, caller_fn_scope, options)[0m
[1;32m    432[0m     [0;32mif[0m [0mis_autograph_strict_conversion_mode[0m[0;34m([0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    433[0m       [0;32mraise[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 434[0;31m     [0;32mreturn[0m [0m_fall_back_unconverted[0m[0;34m([0m[0mf[0m[0;34m,[0m [0margs[0m[0;34m,[0m [0mkwargs[0m[0;34m,[0m [0moptions[0m[0;34m,[0m [0me[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    435[0m [0;34m[0m[0m
[1;32m    436[0m   [0;32mwith[0m [0mStackTraceMapper[0m[0;34m([0m[0mconverted_f[0m[0;34m)[0m[0;34m,[0m [0mtf_stack[0m[0;34m.[0m[0mCurrentModuleFilter[0m[0;34m([0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py[0m in [0;36m_fall_back_unconverted[0;34m(f, args, kwargs, options, exc)[0m
[1;32m    483[0m     [0mlogging[0m[0;34m.[0m[0mwarning[0m[0;34m([0m[0mwarning_template[0m[0;34m,[0m [0mf[0m[0;34m,[0m [0mfile_bug_message[0m[0;34m,[0m [0mexc[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    484[0m [0;34m[0m[0m
[0;32m--> 485[0;31m   [0;32mreturn[0m [0m_call_unconverted[0m[0;34m([0m[0mf[0m[0;34m,[0m [0margs[0m[0;34m,[0m [0mkwargs[0m[0;34m,[0m [0moptions[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    486[0m [0;34m[0m[0m
[1;32m    487[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py[0m in [0;36m_call_unconverted[0;34m(f, args, kwargs, options, update_cache)[0m
[1;32m    458[0m   [0;32mif[0m [0mkwargs[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    459[0m     [0;32mreturn[0m [0mf[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 460[0;31m   [0;32mreturn[0m [0mf[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    461[0m [0;34m[0m[0m
[1;32m    462[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/experimental/image_ops.py[0m in [0;36mdecode_tiff[0;34m(contents, index, name)[0m
[1;32m     85[0m       [0mA[0m[0;31m [0m[0;31m`[0m[0mTensor[0m[0;31m`[0m [0mof[0m [0mtype[0m[0;31m [0m[0;31m`[0m[0muint8[0m[0;31m`[0m [0;32mand[0m [0mshape[0m [0mof[0m[0;31m [0m[0;31m`[0m[0;34m[[0m[0mheight[0m[0;34m,[0m [0mwidth[0m[0;34m,[0m [0;36m4[0m[0;34m][0m[0;31m`[0m [0;34m([0m[0mRGBA[0m[0;34m)[0m[0;34m.[0m[0;34m[0m[0;34m[0m[0m
[1;32m     86[0m     """
[0;32m---> 87[0;31m     [0;32mreturn[0m [0mcore_ops[0m[0;34m.[0m[0mio_decode_tiff[0m[0;34m([0m[0mcontents[0m[0;34m,[0m [0mindex[0m[0;34m,[0m [0mname[0m[0;34m=[0m[0mname[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     88[0m [0;34m[0m[0m
[1;32m     89[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/__init__.py[0m in [0;36m__getattr__[0;34m(self, attrb)[0m
[1;32m     86[0m [0;34m[0m[0m
[1;32m     87[0m     [0;32mdef[0m [0m__getattr__[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mattrb[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 88[0;31m         [0;32mreturn[0m [0mgetattr[0m[0;34m([0m[0mself[0m[0;34m.[0m[0m_load[0m[0;34m([0m[0;34m)[0m[0;34m,[0m [0mattrb[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     89[0m [0;34m[0m[0m
[1;32m     90[0m     [0;32mdef[0m [0m__dir__[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/__init__.py[0m in [0;36m_load[0;34m(self)[0m
[1;32m     82[0m     [0;32mdef[0m [0m_load[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     83[0m         [0;32mif[0m [0mself[0m[0;34m.[0m[0m_mod[0m [0;32mis[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 84[0;31m             [0mself[0m[0;34m.[0m[0m_mod[0m [0;34m=[0m [0m_load_library[0m[0;34m([0m[0mself[0m[0;34m.[0m[0m_library[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     85[0m         [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_mod[0m[0;34m[0m[0;34m[0m[0m
[1;32m     86[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/__init__.py[0m in [0;36m_load_library[0;34m(filename, lib)[0m
[1;32m     67[0m         [0;32mexcept[0m [0;34m([0m[0mtf[0m[0;34m.[0m[0merrors[0m[0;34m.[0m[0mNotFoundError[0m[0;34m,[0m [0mOSError[0m[0;34m)[0m [0;32mas[0m [0me[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     68[0m             [0merrs[0m[0;34m.[0m[0mappend[0m[0;34m([0m[0mstr[0m[0;34m([0m[0me[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 69[0;31m     raise NotImplementedError(
[0m[1;32m     70[0m         [0;34m"unable to open file: "[0m[0;34m[0m[0;34m[0m[0m
[1;32m     71[0m         [0;34m+[0m [0;34mf"{filename}, from paths: {filenames}\ncaused by: {errs}"[0m[0;34m[0m[0;34m[0m[0m

[0;31mNotImplementedError[0m: in user code:

    File "/tmp/ipykernel_11/3550011738.py", line 16, in get_decoded_image  *
        img = tfio.experimental.image.decode_tiff(
    File "/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/experimental/image_ops.py", line 87, in decode_tiff  **
        return core_ops.io_decode_tiff(contents, index, name=name)
    File "/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/__init__.py", line 88, in __getattr__
        return getattr(self._load(), attrb)
    File "/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/__init__.py", line 84, in _load
        self._mod = _load_library(self._library)
    File "/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/__init__.py", line 69, in _load_library
        raise NotImplementedError(

    NotImplementedError: unable to open file: libtensorflow_io.so, from paths: ['/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/libtensorflow_io.so']
    caused by: ['/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/libtensorflow_io.so: undefined symbol: _ZN3tsl8str_util9LowercaseB5cxx11ESt17basic_string_viewIcSt11char_traitsIcEE']


## === cell 22
del X_train, y_train, X_test, y_test
