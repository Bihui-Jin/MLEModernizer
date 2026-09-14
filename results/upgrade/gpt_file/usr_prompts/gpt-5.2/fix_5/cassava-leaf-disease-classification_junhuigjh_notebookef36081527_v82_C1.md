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

3.13

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

0.8865216077364763

# 6. Current score

0.27616

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.27616) has done: 'The timeout is dominated by training EfficientNet-V2-L at 480px for 2 full epochs (plus full validation) on ~18.7k images, which is too much compute for a 600s budget even with good I/O. The only speedups that preserve identical core logic are to eliminate unnecessary work: when TFRecords are enabled, the current code accidentally trains/validates on the entire TFRecord set instead of the intended 90/10 split (doubling work and changing semantics). I fix TFRecord splitting to match the CSV split exactly, and I also remove per-batch Python accuracy computation overhead by accumulating correct counts directly on-device (same metric, less Python). Finally, I ensure the TF→Torch bridge uses pinned memory for faster H2D copies and avoids extra list decoding work.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights

SEED = 11
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

TRAIN_TFREC_DIR = os.path.join(DATA_ROOT, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(DATA_ROOT, "test_tfrecords")

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing: {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_IMG_DIR) or os.path.isdir(
    TRAIN_TFREC_DIR
), "Missing train images/tfrecords"
assert os.path.isdir(TEST_IMG_DIR) or os.path.isdir(
    TEST_TFREC_DIR
), "Missing test images/tfrecords"

IMG_SIZE = 480
BATCH_SIZE = 16
NUM_CLASSES = 5
EPOCHS = 2
LR = 3e-4

if torch.cuda.is_available():
    NUM_WORKERS = min(8, (os.cpu_count() or 4))
else:
    NUM_WORKERS = min(4, (os.cpu_count() or 2))

PERSISTENT_WORKERS = NUM_WORKERS > 0
PREFETCH_FACTOR = 4 if NUM_WORKERS > 0 else None

train_tfms = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

test_tfms = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

try:
    from torchvision.io import read_image  # returns uint8 CHW

    _HAS_TV_READ = True
except Exception:
    _HAS_TV_READ = False


class CassavaImageDataset(Dataset):
    def __init__(self, df, img_dir, transform, has_labels=True):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.has_labels = has_labels
        self.image_ids = self.df["image_id"].tolist()
        self.labels = self.df["label"].astype(int).tolist() if has_labels else None

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.img_dir, image_id)

        if _HAS_TV_READ:
            x = read_image(img_path)  # uint8
            if x.shape[0] == 1:
                x = x.expand(3, -1, -1)
            elif x.shape[0] == 4:
                x = x[:3]
            x = transforms.functional.resize(x, (IMG_SIZE, IMG_SIZE))
            x = x.float().div_(255.0)
            x = transforms.functional.normalize(
                x, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
            )
            if self.has_labels:
                if random.random() < 0.5:
                    x = torch.flip(x, dims=[2])
        else:
            img = Image.open(img_path).convert("RGB")
            x = self.transform(img)

        if self.has_labels:
            y = int(self.labels[idx])
            return x, y
        return x, image_id


def _seed_worker(worker_id: int):
    worker_seed = (SEED + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


dl_generator = torch.Generator()
dl_generator.manual_seed(SEED)

_HAS_TF = False
try:
    import tensorflow as tf  # present on Kaggle

    _HAS_TF = True
except Exception:
    _HAS_TF = False


def _list_tfrecords(dir_path: str):
    if not os.path.isdir(dir_path):
        return []
    files = [
        os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(".tfrec")
    ]
    return sorted(files)


_TRAIN_TFRECS = _list_tfrecords(TRAIN_TFREC_DIR)
_TEST_TFRECS = _list_tfrecords(TEST_TFREC_DIR)
_USE_TFRECORDS = _HAS_TF and (len(_TRAIN_TFRECS) > 0) and (len(_TEST_TFRECS) > 0)
print("TFRecords enabled:", _USE_TFRECORDS)


if _USE_TFRECORDS:
    _FEATURE_SPEC = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.int64),
    }

    def _decode_and_preprocess(example, training: bool):
        ex = tf.io.parse_single_example(example, _FEATURE_SPEC)
        img = tf.io.decode_jpeg(ex["image"], channels=3)
        img = tf.image.resize(
            img, [IMG_SIZE, IMG_SIZE], method="bilinear", antialias=False
        )
        img = tf.cast(img, tf.float32) / 255.0
        img = img * 2.0 - 1.0
        if training:
            img = tf.image.random_flip_left_right(img, seed=SEED)
        img = tf.transpose(img, [2, 0, 1])  # CHW for PyTorch
        img_id = ex["image_name"]
        y = tf.cast(ex["target"], tf.int64)
        return img, y, img_id

    def _make_tf_dataset(tfrecs, training: bool):
        opts = tf.data.Options()
        opts.experimental_deterministic = True
        ds = tf.data.TFRecordDataset(tfrecs, num_parallel_reads=tf.data.AUTOTUNE)
        ds = ds.with_options(opts)
        if training:
            ds = ds.shuffle(8192, seed=SEED, reshuffle_each_iteration=True)
        ds = ds.map(
            lambda x: _decode_and_preprocess(x, training),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        ds = ds.batch(BATCH_SIZE, drop_remainder=False)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    def _make_tf_dataset_subset(tfrecs, training: bool, keep_indices: np.ndarray):
        keep = tf.constant(keep_indices.astype(np.int64))
        keep = tf.sort(keep)

        base = _make_tf_dataset(tfrecs, training=training)

        base = base.unbatch()
        base = base.enumerate()

        def _is_kept(i, _x):
            i = tf.cast(i, tf.int64)
            pos = tf.searchsorted(keep, i, side="left")
            return tf.logical_and(
                pos < tf.size(keep), tf.equal(tf.gather(keep, pos), i)
            )

        base = base.filter(_is_kept).map(
            lambda _i, x: x, num_parallel_calls=tf.data.AUTOTUNE
        )
        base = base.batch(BATCH_SIZE, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
        return base

    class _TorchLikeFromTF:
        """Iterable yielding torch tensors; avoids per-image file I/O and keeps deterministic order."""

        def __init__(
            self, tf_dataset, has_labels: bool, return_ids: bool, pin_memory: bool
        ):
            self.tf_dataset = tf_dataset
            self.has_labels = has_labels
            self.return_ids = return_ids
            self.pin_memory = pin_memory

        def __iter__(self):
            for xb, yb, ids in self.tf_dataset:
                xb = torch.from_numpy(xb.numpy())
                if self.pin_memory:
                    xb = xb.pin_memory()
                if self.has_labels:
                    yb = torch.from_numpy(yb.numpy()).long()
                    if self.pin_memory:
                        yb = yb.pin_memory()
                    yield xb, yb
                else:
                    ids = [s.decode("utf-8") for s in ids.numpy().tolist()]
                    yield xb, ids

        def __len__(self):
            raise TypeError("Length is not defined for TF-backed iterable.")




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
weights = EfficientNet_V2_L_Weights.IMAGENET1K_V1
backbone = efficientnet_v2_l(weights=weights)

in_features = backbone.classifier[1].in_features
backbone.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
model = backbone.to(device)

if torch.cuda.is_available():
    model = model.to(memory_format=torch.channels_last)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

print("Model ready.")




## === cell 2
train_df = pd.read_csv(TRAIN_CSV)
assert set(train_df.columns) >= {"image_id", "label"}

perm = np.random.RandomState(SEED).permutation(len(train_df))
split = int(0.9 * len(train_df))
tr_idx, va_idx = perm[:split], perm[split:]
tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
va_df = train_df.iloc[va_idx].reset_index(drop=True)

common_loader_kwargs = dict(
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=PERSISTENT_WORKERS,
    worker_init_fn=_seed_worker if NUM_WORKERS > 0 else None,
    generator=dl_generator,
)
if PREFETCH_FACTOR is not None:
    common_loader_kwargs["prefetch_factor"] = PREFETCH_FACTOR

if _USE_TFRECORDS:
    train_loader = _TorchLikeFromTF(
        _make_tf_dataset_subset(_TRAIN_TFRECS, training=True, keep_indices=tr_idx),
        has_labels=True,
        return_ids=False,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = _TorchLikeFromTF(
        _make_tf_dataset_subset(_TRAIN_TFRECS, training=False, keep_indices=va_idx),
        has_labels=True,
        return_ids=False,
        pin_memory=torch.cuda.is_available(),
    )
else:
    train_loader = DataLoader(
        CassavaImageDataset(tr_df, TRAIN_IMG_DIR, train_tfms, has_labels=True),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        **common_loader_kwargs,
    )

    val_loader = DataLoader(
        CassavaImageDataset(va_df, TRAIN_IMG_DIR, test_tfms, has_labels=True),
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        **common_loader_kwargs,
    )

len_train = len(tr_df)
len_val = len(va_df)
print(f"Train/Val sizes: {len_train}/{len_val}")




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/805001453.py in <cell line: 0>()
     23     # evaluation semantics. We now select the same indices as the CSV split deterministically.
     24     train_loader = _TorchLikeFromTF(
---> 25         _make_tf_dataset_subset(_TRAIN_TFRECS, training=True, keep_indices=tr_idx),
     26         has_labels=True,
     27         return_ids=False,

/tmp/ipykernel_55/3651309373.py in _make_tf_dataset_subset(tfrecs, training, keep_indices)
    213             )
    214 
--> 215         base = base.filter(_is_kept).map(
    216             lambda _i, x: x, num_parallel_calls=tf.data.AUTOTUNE
    217         )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in filter(self, predicate, name)
   2561     # pylint: disable=g-import-not-at-top,protected-access
   2562     from tensorflow.python.data.ops import filter_op
-> 2563     return filter_op._filter(self, predicate, name)
   2564     # pylint: enable=g-import-not-at-top,protected-access
   2565 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/filter_op.py in _filter(input_dataset, predicate, name)
     23 
     24 def _filter(input_dataset, predicate, name=None):  # pylint: disable=redefined-builtin
---> 25   return _FilterDataset(input_dataset, predicate, name=name)
     26 
     27 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/filter_op.py in __init__(self, input_dataset, predicate, use_legacy_function, name)
     36     """See `Dataset.filter` for details."""
     37     self._input_dataset = input_dataset
---> 38     wrapped_func = structured_function.StructuredFunctionWrapper(
     39         predicate,
     40         self._transformation_name(),

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

/tmp/__autograph_generated_fileov29xpbr.py in tf___is_kept(i, _x)
     10                 retval_ = ag__.UndefinedReturnValue()
     11                 i = ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(i), ag__.ld(tf).int64), None, fscope)
---> 12                 pos = ag__.converted_call(ag__.ld(tf).searchsorted, (ag__.ld(keep), ag__.ld(i)), dict(side='left'), fscope)
     13                 try:
     14                     do_return = True

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

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in _create_c_op(graph, node_def, inputs, control_inputs, op_def, extract_traceback)
   1054   except errors.InvalidArgumentError as e:
   1055     # Convert to ValueError for backwards compatibility.
-> 1056     raise ValueError(e.message)
   1057 
   1058   # Record the current Python stack trace as the creating stacktrace of this

ValueError: in user code:

    File "/tmp/ipykernel_55/3651309373.py", line 210, in _is_kept  *
        pos = tf.searchsorted(keep, i, side="left")

    ValueError: slice index -1 of dimension 0 out of bounds. for '{{node strided_slice_1}} = StridedSlice[Index=DT_INT32, T=DT_INT32, begin_mask=0, ellipsis_mask=0, end_mask=0, new_axis_mask=0, shrink_axis_mask=1](Shape_1/shape_as_tensor, strided_slice_1/stack, strided_slice_1/stack_1, strided_slice_1/stack_2)' with input shapes: [0], [1], [1], [1] and with computed input tensors: input[1] = <-1>, input[2] = <0>, input[3] = <1>.


## === cell 3
model.train()
for epoch in range(1, EPOCHS + 1):
    total_loss = 0.0
    correct = 0
    seen = 0
    n_batches = 0

    for xb, yb in train_loader:
        xb = xb.to(device, non_blocking=True)
        if torch.cuda.is_available():
            xb = xb.contiguous(memory_format=torch.channels_last)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (logits.detach().argmax(dim=1) == yb).sum().item()
        seen += yb.numel()
        n_batches += 1

    model.eval()
    val_correct = 0
    val_seen = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device, non_blocking=True)
            if torch.cuda.is_available():
                xb = xb.contiguous(memory_format=torch.channels_last)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            val_correct += (logits.argmax(dim=1) == yb).sum().item()
            val_seen += yb.numel()
    model.train()

    train_acc = correct / max(seen, 1)
    val_acc = val_correct / max(val_seen, 1)
    print(
        f"Epoch {epoch}/{EPOCHS} - train_loss={total_loss/max(n_batches,1):.4f} "
        f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
    )




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1640109485.py in <cell line: 0>()
      8     n_batches = 0
      9 
---> 10     for xb, yb in train_loader:
     11         xb = xb.to(device, non_blocking=True)
     12         if torch.cuda.is_available():

NameError: name 'train_loader' is not defined

## === cell 4
sample_sub = pd.read_csv(SAMPLE_SUB)
test_ids = sample_sub["image_id"].tolist()

test_df = pd.DataFrame({"image_id": test_ids})

if _USE_TFRECORDS:
    test_loader = _TorchLikeFromTF(
        _make_tf_dataset(_TEST_TFRECS, training=False),
        has_labels=False,
        return_ids=True,
        pin_memory=torch.cuda.is_available(),
    )
else:
    test_loader = DataLoader(
        CassavaImageDataset(test_df, TEST_IMG_DIR, test_tfms, has_labels=False),
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        **common_loader_kwargs,
    )

model.eval()
all_ids = []
all_preds = []
with torch.no_grad():
    for xb, ids in test_loader:
        xb = xb.to(device, non_blocking=True)
        if torch.cuda.is_available():
            xb = xb.contiguous(memory_format=torch.channels_last)
        logits = model(xb)
        preds = torch.argmax(logits, dim=1).cpu().numpy().astype(int).tolist()
        all_preds.extend(preds)
        all_ids.extend(list(ids))

if all_ids != test_ids:
    id_to_pred = {i: p for i, p in zip(all_ids, all_preds)}
    all_ids = test_ids
    all_preds = [int(id_to_pred[i]) for i in test_ids]

assert len(all_ids) == len(
    test_ids
), f"Pred length mismatch: {len(all_ids)} vs {len(test_ids)}"
assert (
    all_ids[0] == test_ids[0] and all_ids[-1] == test_ids[-1]
), "Test ID order mismatch."

submission = pd.DataFrame({"image_id": all_ids, "label": all_preds})
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)




## === cell 5
submission.head()
