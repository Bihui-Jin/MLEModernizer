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

0.8038682381384104

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.14761) has done: 'I fix the runtime failure caused by missing private checkpoint files by switching to a safe fallback: use pretrained timm weights when the expected .pth files are not present, keeping the same two-model ensemble and inference flow. I also correct the input preprocessing to match each backbone’s expected normalization (via timm’s default config) while preserving the same resize/CLAHE core image processing and averaging of logits. Finally, I ensure the submission length and ordering exactly match `sample_submission.csv` (no missing/None paths, no fallback length mismatch) and always write a valid `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import os, sys, glob
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn.functional as F
import timm
import tqdm

import tensorflow as tf

torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True

DATA_DIR = "../input/cassava-leaf-disease-classification"
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
TEST_TFREC_DIR = os.path.join(DATA_DIR, "test_tfrecords")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

assert os.path.exists(
    SAMPLE_SUB_PATH
), f"Missing sample_submission.csv at: {SAMPLE_SUB_PATH}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def _safe_load_state_dict(model, ckpt_path):
    """
    Keep behavior: load checkpoints if present; otherwise fall back to timm pretrained weights.
    """
    if ckpt_path is None or (not os.path.exists(ckpt_path)):
        return False

    sd = torch.load(ckpt_path, map_location="cpu")
    if (
        isinstance(sd, dict)
        and "state_dict" in sd
        and isinstance(sd["state_dict"], dict)
    ):
        sd = sd["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(
        f"Loaded {os.path.basename(ckpt_path)} | missing={len(missing)} unexpected={len(unexpected)}"
    )
    return True


EFF_CKPT = "../input/ensemble-2/eff_model_last.pth"
SERES_CKPT = "../input/ensemble-2/seresnext_model_last.pth"

efficient = timm.create_model("tf_efficientnet_b5", pretrained=True, num_classes=5)
_loaded_eff = _safe_load_state_dict(efficient, EFF_CKPT)
efficient.to(device).eval()

seres = timm.create_model("seresnext101_32x4d", pretrained=True, num_classes=5)
_loaded_seres = _safe_load_state_dict(seres, SERES_CKPT)
seres.to(device).eval()

print(
    f"Models ready. ckpt_loaded: efficient={_loaded_eff}, seresnext={_loaded_seres}\n"
)

eff_cfg = timm.data.resolve_data_config(
    getattr(efficient, "pretrained_cfg", {}), model=efficient
)
ser_cfg = timm.data.resolve_data_config(
    getattr(seres, "pretrained_cfg", {}), model=seres
)

eff_transform = timm.data.create_transform(**eff_cfg, is_training=False)
ser_transform = timm.data.create_transform(**ser_cfg, is_training=False)

print(
    "EfficientNet data cfg:",
    {
        k: eff_cfg[k]
        for k in ["input_size", "interpolation", "mean", "std"]
        if k in eff_cfg
    },
)
print(
    "SEResNeXt data cfg:",
    {
        k: ser_cfg[k]
        for k in ["input_size", "interpolation", "mean", "std"]
        if k in ser_cfg
    },
)



## === cell 2
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def _clahe_bgr_to_rgb_uint8(image_bgr, out_size=(299, 299)):
    """
    Preserve your core logic (resize + CLAHE) but output RGB uint8 for timm transforms.
    """
    img = cv2.resize(image_bgr, out_size)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb


def _timm_preprocess_from_rgb_uint8(img_rgb_uint8, transform):
    """
    Use timm transform (ToTensor + Normalize) to match each backbone's expected preprocessing.
    """
    x = transform(img_rgb_uint8)  # CHW float tensor normalized
    x = x.unsqueeze(0).to(device)
    return x


def _predict_logits_from_bgr(image_bgr):
    """
    Added: simple TTA (horizontal flip) averaged with original logits.
    This improves accuracy while keeping the same models and averaging logits as before.
    """
    img_rgb = _clahe_bgr_to_rgb_uint8(image_bgr, out_size=(299, 299))

    x_eff = _timm_preprocess_from_rgb_uint8(img_rgb, eff_transform)
    x_ser = _timm_preprocess_from_rgb_uint8(img_rgb, ser_transform)
    eff_out = efficient(x_eff)
    ser_out = seres(x_ser)
    logits = (ser_out + eff_out) / 2.0

    img_rgb_fl = np.ascontiguousarray(img_rgb[:, ::-1, :])
    x_eff_fl = _timm_preprocess_from_rgb_uint8(img_rgb_fl, eff_transform)
    x_ser_fl = _timm_preprocess_from_rgb_uint8(img_rgb_fl, ser_transform)
    eff_out_fl = efficient(x_eff_fl)
    ser_out_fl = seres(x_ser_fl)
    logits_fl = (ser_out_fl + eff_out_fl) / 2.0

    return (logits + logits_fl) / 2.0




## === cell 3


def _parse_tfrecord(example_proto):
    image_feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_id": tf.io.FixedLenFeature([], tf.string),
    }
    ex = tf.io.parse_single_example(example_proto, image_feature_description)
    image = tf.io.decode_jpeg(ex["image"], channels=3)
    image_id = ex["image_id"]
    return image, image_id


def _iter_test_from_tfrecords(tfrecord_dir):
    tfrecs = sorted(glob.glob(os.path.join(tfrecord_dir, "*.tfrec")))
    if len(tfrecs) == 0:
        return None, []
    ds = tf.data.TFRecordDataset(tfrecs, num_parallel_reads=tf.data.AUTOTUNE)
    ds = ds.map(_parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    return ds, tfrecs


sample = pd.read_csv(SAMPLE_SUB_PATH)
sample_ids = sample["image_id"].astype(str).tolist()

ds, tfrecs = _iter_test_from_tfrecords(TEST_TFREC_DIR)

pred_map = {}

if ds is not None:
    print(f"Reading test images from TFRecords: {len(tfrecs)} files")
    for image_tf, image_id_tf in tqdm.tqdm(ds, total=sample.shape[0]):
        image_id = image_id_tf.numpy().decode("utf-8")
        img_rgb = image_tf.numpy()
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        logits = _predict_logits_from_bgr(img_bgr)
        pred = int(torch.argmax(logits, dim=1).detach().cpu().item())
        pred_map[image_id] = pred
else:
    print("TFRecords not found; falling back to reading from test_images/*.jpg")
    assert os.path.isdir(TEST_IMG_DIR), f"Missing test_images dir at: {TEST_IMG_DIR}"
    test_files = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg")))
    assert len(test_files) > 0, f"No test images found in: {TEST_IMG_DIR}"
    file_map = {os.path.basename(p): p for p in test_files}

    for image_id in tqdm.tqdm(sample_ids):
        p = file_map.get(image_id, None)
        if p is None:
            pred_map[image_id] = 0
            continue
        img = cv2.imread(p)
        if img is None:
            pred_map[image_id] = 0
            continue
        logits = _predict_logits_from_bgr(img)
        pred_map[image_id] = int(torch.argmax(logits, dim=1).detach().cpu().item())

names, labels = [], []
missing = 0
for img_id in sample_ids:
    names.append(img_id)
    if img_id not in pred_map:
        labels.append(0)
        missing += 1
    else:
        labels.append(int(pred_map[img_id]))

if missing > 0:
    print(
        f"Warning: {missing} image_ids from sample_submission not predicted; set to 0."
    )

assert (
    len(names) == len(sample_ids) == len(labels)
), "Internal length mismatch; submission would be invalid."



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_55/2477321558.py in <cell line: 0>()
     33 if ds is not None:
     34     print(f"Reading test images from TFRecords: {len(tfrecs)} files")
---> 35     for image_tf, image_id_tf in tqdm.tqdm(ds, total=sample.shape[0]):
     36         image_id = image_id_tf.numpy().decode("utf-8")
     37         # TF gives RGB uint8; convert to BGR for the existing CLAHE core that assumes BGR

/usr/local/lib/python3.11/dist-packages/tqdm/std.py in __iter__(self)
   1179 
   1180         try:
-> 1181             for obj in iterable:
   1182                 yield obj
   1183                 # Update and possibly print the progressbar.

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/iterator_ops.py in __next__(self)
    824   def __next__(self):
    825     try:
--> 826       return self._next_internal()
    827     except errors.OutOfRangeError:
    828       raise StopIteration

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/iterator_ops.py in _next_internal(self)
    774     # to communicate that there is no more data to iterate over.
    775     with context.execution_mode(context.SYNC):
--> 776       ret = gen_dataset_ops.iterator_get_next(
    777           self._iterator_resource,
    778           output_types=self._flat_output_types,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/gen_dataset_ops.py in iterator_get_next(iterator, output_types, output_shapes, name)
   3084       return _result
   3085     except _core._NotOkStatusException as e:
-> 3086       _ops.raise_from_not_ok_status(e, name)
   3087     except _core._FallbackException:
   3088       pass

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
   6000 def raise_from_not_ok_status(e, name) -> NoReturn:
   6001   e.message += (" name: " + str(name if name is not None else ""))
-> 6002   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
   6003 
   6004 

InvalidArgumentError: {{function_node __wrapped__IteratorGetNext_output_types_2_device_/job:localhost/replica:0/task:0/device:CPU:0}} Error in user-defined function passed to ParallelMapDatasetV2:2 transformation with iterator: Iterator::Root::ParallelMapV2: Feature: image_id (data type: string) is required but could not be found.
	 [[{{node ParseSingleExample/ParseExample/ParseExampleV2}}]] [Op:IteratorGetNext] name: 

## === cell 4
sub = pd.DataFrame({"image_id": names, "label": labels})

sub["image_id"] = sub["image_id"].astype(str)
sub["label"] = sub["label"].astype(int)

sample = pd.read_csv(SAMPLE_SUB_PATH)
assert (
    sub.shape[0] == sample.shape[0]
), f"Invalid submission length: got {sub.shape[0]}, expected {sample.shape[0]}"
assert list(sub.columns) == [
    "image_id",
    "label",
], f"Invalid submission columns: {sub.columns.tolist()}"

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("Label distribution:\n", sub["label"].value_counts().sort_index())

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2224215572.py in <cell line: 0>()
----> 1 sub = pd.DataFrame({"image_id": names, "label": labels})
      2 
      3 sub["image_id"] = sub["image_id"].astype(str)
      4 sub["label"] = sub["label"].astype(int)
      5 

NameError: name 'names' is not defined
