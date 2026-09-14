# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Detect the presence and position of catheters and lines on chest x-rays.

## Metric
Area under the ROC curve for each label, with the final score being the average of the individual AUCs of each predicted column.

## Submission Format
For each ID in the test set, you must predict a probability for all target variables. The file should contain a header and have the following format:
```
StudyInstanceUID,ETT - Abnormal,ETT - Borderline,ETT - Normal,NGT - Abnormal,NGT - Borderline,NGT - Incompletely Imaged,NGT - Normal,CVC - Abnormal,CVC - Borderline,CVC - Normal,Swan Ganz Catheter Present
1.2.826.0.1.3680043.8.498.62451881164053375557257228990443168843,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.83721761279899623084220697845011427274,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.12732270010839808189235995393981377825,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.11769539755086084996287023095028033598,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.87838627504097587943394933987052577153,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.53211840524738036417560823327351887819,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.93555795394184819372299157360228027866,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.52241894131170494723503100795076463919,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.36500167484503936720548852591033878284,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.86199852603457900780565655267977637728,0,0,0,0,0,0,0,0,0,0,0
```

## Dataset
`train.csv` contains image IDs, binary labels, and patient IDs.

TFRecords are available for both train and test.

We've also included `train_annotations.csv`. These are segmentation annotations for training samples that have them. They are included solely as additional information for competitors.

- train.csv - contains image IDs, binary labels, and patient IDs.
- sample_submission.csv - a sample submission file in the correct format
- test - test images
- train - training images

### Columns
- `StudyInstanceUID` - unique ID for each image
- `ETT - Abnormal` - endotracheal tube placement abnormal
- `ETT - Borderline` - endotracheal tube placement borderline abnormal
- `ETT - Normal` - endotracheal tube placement normal
- `NGT - Abnormal` - nasogastric tube placement abnormal
- `NGT - Borderline` - nasogastric tube placement borderline abnormal
- `NGT - Incompletely Imaged` - nasogastric tube placement inconclusive due to imaging
- `NGT - Normal` - nasogastric tube placement borderline normal
- `CVC - Abnormal` - central venous catheter placement abnormal
- `CVC - Borderline` - central venous catheter placement borderline abnormal
- `CVC - Normal` - central venous catheter placement normal
- `Swan Ganz Catheter Present`
- `PatientID` - unique ID for each patient in the dataset

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (172 lines)
            sample_submission.csv (3010 lines)
            sample_submission.csv.zip (64.2 kB)
            test.zip (642.8 MB)
            train.csv (27075 lines)
            train.csv.zip (798.6 kB)
            train.zip (5.8 GB)
            train_annotations.csv (16262 lines)
            train_annotations.csv.zip (1.4 MB)
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
            test/
                1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                ... and 3007 other files
                test/
            train/
                1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                ... and 27072 other files
                train/
        input/
            description.md (172 lines)
            sample_submission.csv (3010 lines)
            sample_submission.csv.zip (64.2 kB)
            test.zip (642.8 MB)
            train.csv (27075 lines)
            train.csv.zip (798.6 kB)
            train.zip (5.8 GB)
            train_annotations.csv (16262 lines)
            train_annotations.csv.zip (1.4 MB)
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
            test/
                1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                ... and 3007 other files
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
            train/
                1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                ... and 27072 other files
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
        working/
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
```

-> data/ranzcr-clip-catheter-line-classification/sample_submission.csv has 3009 rows and 10 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline

-> data/ranzcr-clip-catheter-line-classification/train.csv has 27074 rows and 13 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline, CVC - Normal, Swan Ganz Catheter Present, PatientID

-> data/ranzcr-clip-catheter-line-classification/train_annotations.csv has 16261 rows and 3 columns.
The columns are: StudyInstanceUID, label, data

-> data/sample_submission.csv has 3009 rows and 10 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline

-> data/train.csv has 27074 rows and 13 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline, CVC - Normal, Swan Ganz Catheter Present, PatientID

-> data/train_annotations.csv has 16261 rows and 3 columns.
The columns are: StudyInstanceUID, label, data

-> (stopped after 10 files for performance)

# 5. Target score

0.7393840001637286

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I add the missing imports (`numpy` and `pandas`), adjust the test‑image count to use the length of the loaded submission (ensuring a non‑zero count), and keep the rest of the logic unchanged so the script runs end‑to‑end and writes a valid `submission.csv`.'
- What this solution (achieved 0.5) has done: 'I make the TensorFlow‑related functions safely no‑op when TensorFlow cannot be initialized, so the script runs without errors on the CPU‑only environment. This fixes the runtime crash while keeping the original logic unchanged; the baseline prediction (label means) is still used, which moves the score away from the pure‑random 0.5 baseline toward the target.'
- What this solution (achieved 0.5) has done: 'The changes add the missing imports, define undefined globals (`REPLICAS`, `AUTO`), safely handle TensorFlow availability, and keep the original baseline logic intact so the script runs end‑to‑end and writes a correct `submission.csv` file.'
- What this solution (achieved 0.5) has done: 'We parallelize image loading and replace the slow `iterrows` loop with vectorized indexing, while keeping the same sampling, resizing, and logistic‑regression logic. By using a thread pool we reduce I/O wait time for the ~8 k images (training + test) and avoid unnecessary `os.path.exists` checks, which brings total runtime well under the 600 s limit without altering model behavior.'
- What this solution (achieved 0.66391) has done: 'We flatten the image arrays before feeding them to scikit‑learn, because LogisticRegression expects a 2‑D feature matrix. This small change lets the model train instead of falling back to the baseline, moving the AUC closer to the target while keeping all original logic untouched.'
- What this solution (achieved 0.66109) has done: 'The main slowdown comes from importing TensorFlow, which is never used because `tf` is forced to `None`. By skipping that heavy import we avoid the long startup cost, while all other logic—including the logistic‑regression model and image handling—remains unchanged.'
- What this solution (achieved 0.5) has done: 'Implemented a lightweight validation split to tune a blending weight between the logistic‑regression model predictions and the simple label‑mean baseline.  
The script now:
1. Splits the sampled training set into train/validation parts.  
2. Trains the One‑Vs‑Rest LogisticRegression on the training split.  
3. Searches for the blending weight `w` (0‑1) that maximises the mean AUC on the validation split.  
4. Applies this optimal `w` to combine model‐based probabilities with the baseline for the final test predictions.  
This modest adjustment is expected to move the AUC from ~0.66 toward the target ~0.74 while preserving the original pipeline.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os, glob, random
from PIL import Image

tf = None

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
except Exception:
    LogisticRegression = None
    OneVsRestClassifier = None
    train_test_split = None
    roc_auc_score = None

REPLICAS = 1  # single‑replica execution
BATCH_SIZE = 16 * REPLICAS
HEIGHT = 512
WIDTH = 512
CHANNELS = 3
N_CLASSES = 5
TTA_STEPS = 3  # Do TTA if > 0
IMAGE_SIZE = [512, 512]  # At this size, a GPU will run out of memory. Use the TPU.
SEED = 555
AUG_BATCH = BATCH_SIZE
AUTO = None  # No TensorFlow, so no AUTOTUNE




## === cell 1
def data_augment(image, label):
    if tf is None:
        return image, label
    image = tf.image.rot90(image, k=np.random.randint(4))
    image = tf.image.random_flip_left_right(image, seed=SEED)
    image = tf.image.random_flip_up_down(image, seed=SEED)
    IMG_SIZE = IMAGE_SIZE[0]
    image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE + 6, IMG_SIZE + 6)
    image = tf.image.random_crop(image, size=[IMG_SIZE, IMG_SIZE, 3])
    image = tf.image.random_brightness(image, max_delta=0.5)
    image = tf.image.random_saturation(image, 0, 2, seed=SEED)
    image = tf.image.adjust_saturation(image, 3)
    return image, label


def to_float32_2(image, label):
    if tf is None:
        img_np = np.asarray(image, dtype=np.float32)
        lbl_np = np.asarray(label, dtype=np.int32)
        return img_np, lbl_np
    max_val = tf.reduce_max(label, axis=-1, keepdims=True)
    cond = tf.equal(label, max_val)
    label = tf.where(cond, tf.ones_like(label), tf.zeros_like(label))
    return tf.cast(image, tf.float32), tf.cast(label, tf.int32)


def to_float32(image, label):
    if tf is None:
        img_np = np.asarray(image, dtype=np.float32)
        return img_np, label
    return tf.cast(image, tf.float32), label


def decode_image(image_data):
    if tf is None:
        return np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.float32)
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [1024, 1024, 3])
    return image


def read_labeled_tfrecord(example):
    if tf is None:
        raise RuntimeError("TensorFlow is required to read TFRecords.")
    train_feature_description = {
        "CVC - Abnormal": tf.io.FixedLenFeature([], tf.int64),
        "CVC - Borderline": tf.io.FixedLenFeature([], tf.int64),
        "CVC - Normal": tf.io.FixedLenFeature([], tf.int64),
        "ETT - Abnormal": tf.io.FixedLenFeature([], tf.int64),
        "ETT - Borderline": tf.io.FixedLenFeature([], tf.int64),
        "ETT - Normal": tf.io.FixedLenFeature([], tf.int64),
        "NGT - Abnormal": tf.io.FixedLenFeature([], tf.int64),
        "NGT - Borderline": tf.io.FixedLenFeature([], tf.int64),
        "NGT - Incompletely Imaged": tf.io.FixedLenFeature([], tf.int64),
        "NGT - Normal": tf.io.FixedLenFeature([], tf.int64),
        "StudyInstanceUID": tf.io.FixedLenFeature([], tf.string),
        "Swan Ganz Catheter Present": tf.io.FixedLenFeature([], tf.int64),
        "image": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, train_feature_description)
    image = decode_image(example["image"])
    values = [
        example["ETT - Abnormal"],
        example["ETT - Borderline"],
        example["ETT - Normal"],
        example["NGT - Abnormal"],
        example["NGT - Borderline"],
        example["NGT - Incompletely Imaged"],
        example["NGT - Normal"],
        example["CVC - Abnormal"],
        example["CVC - Borderline"],
        example["CVC - Normal"],
        example["Swan Ganz Catheter Present"],
    ]
    label = tf.cast(0, tf.int32)
    for i in range(len(values)):
        if values[i] == 1:
            label = tf.cast(i, tf.int32)
    return image, label


def read_unlabeled_tfrecord(example):
    if tf is None:
        raise RuntimeError("TensorFlow is required to read TFRecords.")
    UNLABELED_TFREC_FORMAT = {
        "StudyInstanceUID": tf.io.FixedLenFeature([], tf.string),
        "image": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)
    image = decode_image(example["image"])
    image = tf.image.resize(image, [IMAGE_SIZE[0], IMAGE_SIZE[0]])
    image_name = example["StudyInstanceUID"]
    return image, image_name


def load_dataset(filenames, labeled=True, ordered=False):
    if tf is None:
        raise RuntimeError("TensorFlow is required to load datasets.")
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(
        read_labeled_tfrecord if labeled else read_unlabeled_tfrecord,
        num_parallel_calls=AUTO,
    )
    return dataset


def get_training_dataset(dataset, do_aug=True, do_onehot=False):
    if tf is None:
        raise RuntimeError("TensorFlow is required for training dataset pipeline.")
    if do_aug:
        dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
    dataset = dataset.repeat()
    dataset = dataset.batch(AUG_BATCH)
    if do_onehot:
        dataset = dataset.map(onehot, num_parallel_calls=AUTO)
    dataset = dataset.unbatch()
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO)
    return dataset


def get_validation_dataset(filenames, ordered=False):
    if tf is None:
        raise RuntimeError("TensorFlow is required for validation dataset pipeline.")
    dataset = load_dataset(filenames, labeled=True, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.cache()
    dataset = dataset.prefetch(AUTO)
    return dataset


def get_test_dataset(filenames, ordered=False, tta=False):
    if tf is None:
        raise RuntimeError("TensorFlow is required for test dataset pipeline.")
    dataset = load_dataset(filenames, labeled=False, ordered=ordered)
    if tta:
        dataset = dataset.map(
            lambda img, name: (data_augment(img, tf.zeros_like(img))[0], name),
            num_parallel_calls=AUTO,
        )
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO)
    return dataset


def count_data_items(filenames):
    if not filenames:
        return 0
    return len(filenames)




## === cell 2
import concurrent.futures

database_base_path = "/kaggle/input/ranzcr-clip-catheter-line-classification/"

submission = pd.read_csv(os.path.join(database_base_path, "sample_submission.csv"))
label_cols = [c for c in submission.columns if c != "StudyInstanceUID"]
print(f"Target columns ({len(label_cols)}): {label_cols}")

train_df = pd.read_csv(os.path.join(database_base_path, "train.csv"))
train_df = train_df[["StudyInstanceUID"] + label_cols]  # keep only needed columns

label_means = train_df[label_cols].mean().values.astype(np.float32)

use_model = False
best_w = 1.0  # fallback: pure model if we cannot compute a blend

if (
    LogisticRegression is not None
    and OneVsRestClassifier is not None
    and train_test_split is not None
):
    try:
        IMG_SIZE = (64, 64)  # small enough for quick training
        MAX_TRAIN_SAMPLES = 8000  # a bit larger for better representation
        random.seed(SEED)

        if len(train_df) > MAX_TRAIN_SAMPLES:
            sampled_df = train_df.sample(n=MAX_TRAIN_SAMPLES, random_state=SEED)
        else:
            sampled_df = train_df

        uids_all = sampled_df["StudyInstanceUID"].values
        labels_all = sampled_df[label_cols].values.astype(np.int32)

        def load_vec(uid):
            path = os.path.join(database_base_path, "train", f"{uid}.jpg")
            try:
                img = Image.open(path).convert("RGB")
                img = img.resize(IMG_SIZE, Image.BILINEAR)
                return np.asarray(img, dtype=np.float32) / 255.0
            except Exception:
                return None

        X_list = []
        y_list = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            for idx, vec in enumerate(executor.map(load_vec, uids_all)):
                if vec is not None:
                    X_list.append(vec)
                    y_list.append(labels_all[idx])

        if not X_list:
            raise RuntimeError("No training images loaded.")

        X = np.stack(X_list).reshape(len(X_list), -1)  # (n, 12288)
        y = np.stack(y_list)

        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.2, random_state=SEED, stratify=y
        )

        clf = OneVsRestClassifier(
            LogisticRegression(
                max_iter=500, solver="liblinear", class_weight="balanced"
            )
        )
        clf.fit(X_tr, y_tr)

        val_proba = clf.predict_proba(X_val)  # shape (n_val, n_labels)

        baseline_val = np.tile(label_means, (X_val.shape[0], 1))

        best_auc = -1.0
        for w in np.linspace(0, 1, 11):
            blended = w * val_proba + (1 - w) * baseline_val
            aucs = []
            for i in range(len(label_cols)):
                try:
                    auc = roc_auc_score(y_val[:, i], blended[:, i])
                    aucs.append(auc)
                except Exception:
                    continue
            if aucs:
                mean_auc = np.mean(aucs)
                if mean_auc > best_auc:
                    best_auc = mean_auc
                    best_w = w
        print(
            f"Best blending weight on validation: {best_w:.2f} (mean AUC {best_auc:.4f})"
        )

        use_model = True
        X_test_template = None  # placeholder, will be recomputed later
        print(
            f"Trained logistic regression on {X_tr.shape[0]} samples (validation set size {X_val.shape[0]})."
        )
    except Exception as e:
        print(f"Model training / validation failed ({e}); using baseline only.")
        use_model = False
        best_w = 0.0  # fallback to pure baseline
else:
    print("scikit‑learn not available; using baseline.")
    best_w = 0.0  # pure baseline

test_ids = submission["StudyInstanceUID"].values.astype(str)

if use_model:
    def load_vec_test(uid):
        for sub in ("train", "test"):
            path = os.path.join(database_base_path, sub, f"{uid}.jpg")
            try:
                img = Image.open(path).convert("RGB")
                img = img.resize(IMG_SIZE, Image.BILINEAR)
                return np.asarray(img, dtype=np.float32) / 255.0
            except Exception:
                continue
        return None

    X_test_list = []
    valid_test_ids = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for uid, vec in zip(test_ids, executor.map(load_vec_test, test_ids)):
            if vec is not None:
                X_test_list.append(vec)
                valid_test_ids.append(uid)

    if X_test_list:
        X_test = np.stack(X_test_list).reshape(len(X_test_list), -1)
        model_proba = clf.predict_proba(X_test)  # (n_valid, n_labels)

        probabilities = np.tile(label_means, (len(test_ids), 1))
        id_to_index = {uid: i for i, uid in enumerate(valid_test_ids)}
        for i, uid in enumerate(test_ids):
            if uid in id_to_index:
                probabilities[i] = model_proba[id_to_index[uid]]
        probabilities = best_w * probabilities + (1 - best_w) * np.tile(
            label_means, (len(test_ids), 1)
        )
        print("Model predictions generated and blended.")
    else:
        print("No test images loaded; using baseline probabilities.")
        probabilities = np.tile(label_means, (len(test_ids), 1))
else:
    probabilities = np.tile(label_means, (len(test_ids), 1))

print(f"Probabilities shape: {probabilities.shape}")



## === cell 3
print("Generating submission.csv file...")
if probabilities.shape[0] != len(test_ids):
    if probabilities.shape[0] > len(test_ids):
        probabilities = probabilities[: len(test_ids), :]
    else:
        pad_rows = len(test_ids) - probabilities.shape[0]
        pad = np.tile(label_means, (pad_rows, 1))
        probabilities = np.vstack([probabilities, pad])

df_submission = pd.DataFrame(probabilities, columns=label_cols)
df_submission.insert(0, "StudyInstanceUID", test_ids)
df_submission.to_csv("submission.csv", index=False)
print("submission.csv written successfully.")
