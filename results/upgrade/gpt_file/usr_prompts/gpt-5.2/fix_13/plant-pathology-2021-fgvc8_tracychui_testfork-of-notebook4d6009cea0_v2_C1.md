# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import glob
import gc
import numpy as np
import pandas as pd

SEED = 123
np.random.seed(SEED)

os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=2")

TF_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import backend as K
    import tensorflow.keras.applications.resnet50 as resnet

    TF_AVAILABLE = True
    K.set_image_data_format("channels_last")
    tf.random.set_seed(SEED)

    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    try:
        tf.config.optimizer.set_jit(True)
    except Exception:
        pass

    try:
        tf.config.threading.set_intra_op_parallelism_threads(0)
        tf.config.threading.set_inter_op_parallelism_threads(0)
    except Exception:
        pass

    try:
        tf.data.experimental.enable_debug_mode(False)
    except Exception:
        pass

    print(
        "Using TensorFlow backend:", "keras:", keras.__version__, "tf:", tf.__version__
    )
except Exception as e:
    print(
        "TensorFlow import failed, will fall back to PyTorch/torchvision for feature extraction."
    )
    print("TF error:", repr(e))
    TF_AVAILABLE = False

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression



## === cell 1
IMG_WIDTH = 300
IMG_HEIGHT = 300
NR_CHANNELS = 3

TRAIN_DIR = "../input/plant-pathology-2021-fgvc8/train_images"
TEST_DIR = "../input/plant-pathology-2021-fgvc8/test_images"
TRAIN_CSV = "../input/plant-pathology-2021-fgvc8/train.csv"
SAMPLE_SUB = "../input/plant-pathology-2021-fgvc8/sample_submission.csv"



## === cell 2
imglist_test = sorted(glob.glob(os.path.join(TEST_DIR, "*.jpg")))
print("n_test_images:", len(imglist_test))
assert len(imglist_test) > 0, "No test images found - check TEST_DIR path."



## === cell 3
training_csv = pd.read_csv(TRAIN_CSV)

all_tokens = training_csv["labels"].fillna("").astype(str).str.split()
tagnames = np.unique(np.concatenate(all_tokens.to_list()))
tagnames = tagnames[tagnames != ""]  # drop possible empty token from missing labels
print("n_train:", len(training_csv), "n_tags:", len(tagnames))
print("tags:", tagnames)




## === cell 4
def labels_to_multihot(labels_series, tagnames):
    from sklearn.preprocessing import MultiLabelBinarizer

    mlb = MultiLabelBinarizer(classes=list(tagnames))
    split = labels_series.fillna("").astype(str).str.split().tolist()
    Y = mlb.fit_transform(split).astype(np.int8, copy=False)
    return Y


Y = labels_to_multihot(training_csv["labels"], tagnames)



## === cell 5
train_paths = [os.path.join(TRAIN_DIR, fn) for fn in training_csv["image"].tolist()]
print("n_train_after_filter:", len(train_paths), "Y shape:", Y.shape)



## === cell 6
feat_model = None
feat_dim = None

if TF_AVAILABLE:
    base = resnet.ResNet50(
        weights="imagenet", include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    )
    inp = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = base(inp, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    feat_model = keras.Model(inp, x)
    feat_dim = feat_model.output_shape[-1]

    try:
        feat_model.compile(steps_per_execution=32)
    except Exception:
        pass

    @tf.function(reduce_retracing=True)
    def _decode_resize_preprocess(path):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img_bytes, channels=3)
        img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH], method="bilinear")
        img = tf.cast(img, tf.float32)
        img = resnet.preprocess_input(img)
        img.set_shape([IMG_HEIGHT, IMG_WIDTH, 3])
        return img

    print("TF feature dim:", feat_dim)
else:
    import torch
    import torchvision
    from torchvision import transforms
    from PIL import Image

    torch.manual_seed(SEED)

    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
    base = torchvision.models.resnet50(weights=weights)
    base = torch.nn.Sequential(*(list(base.children())[:-1]))  # drop fc, keep avgpool
    base.eval()

    preprocess = transforms.Compose(
        [
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=weights.transforms().mean, std=weights.transforms().std
            ),
        ]
    )

    feat_dim = 2048
    print("Torch feature dim:", feat_dim)




## === cell 7
def extract_features(paths, batch_size=32):
    if TF_AVAILABLE:
        paths_tf = tf.constant(paths)
        ds = tf.data.Dataset.from_tensor_slices(paths_tf)

        opts = tf.data.Options()
        try:
            opts.experimental_deterministic = True
            opts.experimental_optimization.apply_default_optimizations = True
            opts.experimental_optimization.map_parallelization = True
            opts.experimental_optimization.parallel_batch = True
            opts.experimental_optimization.autotune_buffers = True
        except Exception:
            pass
        ds = ds.with_options(opts)

        drop = (len(paths) % batch_size) == 0

        ds = ds.map(
            _decode_resize_preprocess,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True,
        )
        ds = ds.batch(batch_size, drop_remainder=drop)
        ds = ds.prefetch(tf.data.AUTOTUNE)

        feats = feat_model.predict(ds, verbose=0)
        return np.asarray(feats, dtype=np.float32)

    else:
        import torch
        from PIL import Image
        from torch.utils.data import Dataset, DataLoader

        class _ImgPathDataset(Dataset):
            def __init__(self, paths_):
                self.paths = paths_

            def __len__(self):
                return len(self.paths)

            def __getitem__(self, idx):
                p = self.paths[idx]
                img = Image.open(p).convert("RGB")
                return preprocess(img)

        device = torch.device("cpu")
        ds = _ImgPathDataset(paths)

        num_workers = min(4, (os.cpu_count() or 2))
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=False,
        )

        feats_out = np.empty((len(paths), feat_dim), dtype=np.float32)
        base_cpu = base

        with torch.no_grad():
            offset = 0
            for bt in loader:
                bt = bt.to(device, non_blocking=False)
                f = base_cpu(bt)  # (B, 2048, 1, 1)
                f = f.view(f.shape[0], -1).cpu().numpy().astype(np.float32, copy=False)
                bsz = f.shape[0]
                feats_out[offset : offset + bsz] = f
                offset += bsz

        return feats_out




## === cell 8
Xf_train = extract_features(train_paths, batch_size=256)
print("Xf_train:", Xf_train.shape)



## === cell 9
Xf_test = extract_features(imglist_test, batch_size=256)
print("Xf_test:", Xf_test.shape)
gc.collect()



## === cell 10
scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
trainXn = scaler.fit_transform(Xf_train)
testXn_test = scaler.transform(Xf_test)
print("trainXn:", trainXn.shape, "testXn_test:", testXn_test.shape)



## === cell 11
col_sums = Y.sum(axis=0)
is_const = (col_sums == 0) | (col_sums == len(Y))
nonconst_idx = np.where(~is_const)[0]

multi_lr = None
if len(nonconst_idx) > 0:
    multi_lr = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
    multi_lr.fit(trainXn, Y[:, nonconst_idx])

print("trained tag models:", int(len(nonconst_idx)), "/", len(tagnames))



## === cell 12
testKaggle_ppredscore1 = np.zeros((len(testXn_test), len(tagnames)), dtype=np.float32)

const_cols = np.where(is_const)[0]
if const_cols.size:
    const_vals = (col_sums[const_cols] == len(Y)).astype(np.float32)
    testKaggle_ppredscore1[:, const_cols] = const_vals

if multi_lr is not None:
    prob = multi_lr.predict_proba(testXn_test)
    if isinstance(prob, list):
        pos_probs = np.column_stack([p[:, 1] for p in prob]).astype(
            np.float32, copy=False
        )
    else:
        pos_probs = prob[:, :, 1].astype(np.float32, copy=False)
    testKaggle_ppredscore1[:, nonconst_idx] = pos_probs

print("pred score matrix:", testKaggle_ppredscore1.shape)




## === cell 13
def class2tags(classes, tagnames):
    tagnames = np.asarray(tagnames)
    return [" ".join(tagnames[row]) for row in classes]




## === cell 14
test_predclass = testKaggle_ppredscore1 > 0.5
test_predtags = class2tags(test_predclass, tagnames)

for i in range(len(test_predtags)):
    if test_predtags[i] == "":
        test_predtags[i] = "healthy"

print("example preds:", test_predtags[:5])



## === cell 15
sample = pd.read_csv(SAMPLE_SUB)
assert list(sample.columns) == ["image", "labels"]

pred_df = pd.DataFrame(
    {"image": [os.path.basename(p) for p in imglist_test], "labels": test_predtags}
)

submission = sample[["image"]].merge(pred_df, on="image", how="left")
submission["labels"] = submission["labels"].fillna("healthy")

assert list(submission.columns) == ["image", "labels"]
assert len(submission) == len(
    sample
), f"Submission rows {len(submission)} != sample rows {len(sample)}"

submission.to_csv("./submission.csv", index=False)
print("Wrote ./submission.csv with shape:", submission.shape)
print(submission.head())
