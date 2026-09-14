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
Predict whether a lesion is malignant (0 denotes **benign**, and 1 indicates **malignant**).

## Metric
Area under the ROC curve.

## Submission Format
For each `image_name` in the test set, you must predict the probability (`target`) that the sample is **malignant**. The file should contain a header and have the following format:

```
image_name,target
ISIC_0052060,0.7
ISIC_0052349,0.9
ISIC_0058510,0.8
ISIC_0073313,0.5
ISIC_0073502,0.5
etc.
```

## Dataset 
The images are provided in DICOM format.

Images are also provided in JPEG and TFRecord format (in the `jpeg` and `tfrecords` directories, respectively). Images in TFRecord format have been resized to a uniform 1024x1024.

Metadata is also provided outside of the DICOM format, in CSV files. See the `Columns` section for a description.

### Files
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

### Columns
- `image_name` - unique identifier, points to filename of related DICOM image
- `patient_id` - unique patient identifier
- `sex` - the sex of the patient (when unknown, will be blank)
- `age_approx` - approximate patient age at time of imaging
- `anatom_site_general_challenge` - location of imaged site
- `diagnosis` - detailed diagnosis information (train only)
- `benign_malignant` - indicator of malignancy of imaged lesion
- `target` - binarized version of the target variable

# 2. Python version

3.8

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (176 lines)
            jpeg.zip (24.7 GB)
            sample_submission.csv (4143 lines)
            sample_submission.csv.zip (16.4 kB)
            test.csv (4143 lines)
            test.csv.zip (42.5 kB)
            test.zip (6.4 GB)
            tfrecords.zip (9.3 GB)
            train.csv (28985 lines)
            train.csv.zip (299.7 kB)
            train.zip (46.0 GB)
            jpeg/
                test/
                    ISIC_1440063.jpg (1.1 MB)
                    ISIC_0815802.jpg (853.1 kB)
                    ... and 4140 other files
                train/
                    ISIC_1845271.jpg (1.0 MB)
                    ISIC_1970027.jpg (138.4 kB)
                    ... and 28982 other files
            siim-isic-melanoma-classification/
                description.md (176 lines)
                jpeg.zip (24.7 GB)
                ... and 9 other files
                jpeg/
                    test/
                        ISIC_1440063.jpg (1.1 MB)
                        ISIC_0815802.jpg (853.1 kB)
                        ... and 4140 other files
                    train/
                        ISIC_1845271.jpg (1.0 MB)
                        ISIC_1970027.jpg (138.4 kB)
                        ... and 28982 other files
                siim-isic-melanoma-classification/
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
                tfrecords/
                    test00-2071.tfrec (579.6 MB)
                    test01-2071.tfrec (583.5 MB)
                    ... and 14 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
            test/
                ISIC_0052212.dcm (1.5 MB)
                ISIC_0076545.dcm (4.0 MB)
                ... and 4140 other files
                test/
            tfrecords/
                test00-2071.tfrec (579.6 MB)
                test01-2071.tfrec (583.5 MB)
                ... and 14 other files
            train/
                ISIC_0015719.dcm (2.4 MB)
                ISIC_0068279.dcm (1.3 MB)
                ... and 28982 other files
                train/
        input/
            description.md (176 lines)
            jpeg.zip (24.7 GB)
            sample_submission.csv (4143 lines)
            sample_submission.csv.zip (16.4 kB)
            test.csv (4143 lines)
            test.csv.zip (42.5 kB)
            test.zip (6.4 GB)
            tfrecords.zip (9.3 GB)
            train.csv (28985 lines)
            train.csv.zip (299.7 kB)
            train.zip (46.0 GB)
            jpeg/
                test/
                    ISIC_1440063.jpg (1.1 MB)
                    ISIC_0815802.jpg (853.1 kB)
                    ... and 4140 other files
                train/
                    ISIC_1845271.jpg (1.0 MB)
                    ISIC_1970027.jpg (138.4 kB)
                    ... and 28982 other files
            siim-isic-melanoma-classification/
                description.md (176 lines)
                jpeg.zip (24.7 GB)
                ... and 9 other files
                jpeg/
                    test/
                        ISIC_1440063.jpg (1.1 MB)
                        ISIC_0815802.jpg (853.1 kB)
                        ... and 4140 other files
                    train/
                        ISIC_1845271.jpg (1.0 MB)
                        ISIC_1970027.jpg (138.4 kB)
                        ... and 28982 other files
                siim-isic-melanoma-classification/
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
                tfrecords/
                    test00-2071.tfrec (579.6 MB)
                    test01-2071.tfrec (583.5 MB)
                    ... and 14 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
            test/
                ISIC_0052212.dcm (1.5 MB)
                ISIC_0076545.dcm (4.0 MB)
                ... and 4140 other files
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
            tfrecords/
                test00-2071.tfrec (579.6 MB)
                test01-2071.tfrec (583.5 MB)
                ... and 14 other files
            train/
                ISIC_0015719.dcm (2.4 MB)
                ISIC_0068279.dcm (1.3 MB)
                ... and 28982 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
        working/
            siim-isic-melanoma-classification/
                description.md (176 lines)
                jpeg.zip (24.7 GB)
                ... and 9 other files
                jpeg/
                    test/
                        ISIC_1440063.jpg (1.1 MB)
                        ISIC_0815802.jpg (853.1 kB)
                        ... and 4140 other files
                    train/
                        ISIC_1845271.jpg (1.0 MB)
                        ISIC_1970027.jpg (138.4 kB)
                        ... and 28982 other files
                siim-isic-melanoma-classification/
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
                tfrecords/
                    test00-2071.tfrec (579.6 MB)
                    test01-2071.tfrec (583.5 MB)
                    ... and 14 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
```

-> data/sample_submission.csv has 4142 rows and 2 columns.
The columns are: image_name, target

-> data/siim-isic-melanoma-classification/sample_submission.csv has 4142 rows and 2 columns.
The columns are: image_name, target

-> data/siim-isic-melanoma-classification/test.csv has 4142 rows and 5 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge

-> data/siim-isic-melanoma-classification/train.csv has 28984 rows and 8 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge, diagnosis, benign_malignant, target

-> data/test.csv has 4142 rows and 5 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge

-> data/train.csv has 28984 rows and 8 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge, diagnosis, benign_malignant, target

-> input/sample_submission.csv has 4142 rows and 2 columns.
The columns are: image_name, target

-> (stopped after 10 files for performance)

# 5. Target score

0.8687963943891686

# 6. Current score

0.3584

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.57536) has done: 'I fix the immediate runtime failure caused by an environment-level protobuf/TensorFlow import issue by pinning TensorFlow’s protobuf implementation to the pure-Python backend before importing TensorFlow. Then I remove the hard dependency on an external weights file that is not present in your provided `/kaggle/input/...` paths by switching to EfficientNetB0’s built-in ImageNet weights (same architecture) so inference can run end-to-end. Finally, I keep your TTA/inference pipeline intact and ensure the submission is aligned to `sample_submission.csv` ordering so `submission.csv` is always valid and correctly formatted.'
- What this solution (achieved 0.56393) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf backend *and* disabling the C++ one explicitly, and I also make the TensorFlow import more robust by setting these env vars before any TF-related import happens. Next, I load the EfficientNetB0 ImageNet weights as before but also add the missing compilation/weight loading logic check so we never run with random weights unintentionally (which is the main reason the AUC is far below target). Finally, I keep your TFRecord + TTA pipeline intact while ensuring deterministic ID/pred alignment by deriving `test_ids` from the same finite, non-repeated dataset slice used for predictions, so the submission rows correctly match the averaged predictions.'
- What this solution (achieved 0.50358) has done: 'You’re crashing before any training/inference because the Kaggle TensorFlow build is hitting a protobuf incompatibility (`MessageFactory.GetPrototype`). I fix this by forcing the pure-Python protobuf implementation *and* disabling the C++ backend before TensorFlow is imported, which avoids that specific attribute error in this environment. Then, to move the score toward your target (your current AUC strongly suggests random/untrained head weights), I minimally load a standard pretrained EfficientNetB0 classification checkpoint (same architecture family) and adapt it to your 1-output sigmoid head by copying the backbone weights and leaving only the final Dense randomly initialized. Finally, I keep your TFRecord + TTA pipeline intact but make ID/pred alignment deterministic by generating `test_ids` from the exact same non-repeated base dataset used for counting and prediction ordering.'
- What this solution (achieved 0.52677) has done: 'We fix the immediate TensorFlow/protobuf import crash by setting the protobuf environment variables *before* Python imports the `google.protobuf` module, and by forcing TensorFlow to use the Python protobuf implementation in a way that actually takes effect in Kaggle. Then we keep your TFRecord + TTA inference pipeline intact but make sure the model is properly initialized for inference by compiling it (score-neutral) and ensuring pretrained backbone weights are successfully loaded (otherwise predictions can collapse toward random). Finally, we keep the submission alignment/merge with `sample_submission.csv` and always write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.50643) has done: 'I fix the TensorFlow/protobuf import crash by ensuring the pure-Python protobuf backend is selected *before* any protobuf/TensorFlow import and by removing the early `google.protobuf` import that can lock in the wrong implementation. I also add a safe fallback path to use the duplicate dataset location under `/kaggle/data/...` if `/kaggle/input/...` isn’t the one actually mounted in your runtime, so TFRecords and sample submission are always found. Finally, I keep your model/TTA pipeline intact but make the prediction/ID ordering consistent by deriving `test_ids` from the exact same repeated/augmented stream used for predictions (rather than a potentially differently-ordered base dataset), which should legitimately improve AUC toward the target without changing the core approach.'
- What this solution (achieved 0.3584) has done: 'I fix the immediate TensorFlow/protobuf crash by forcing the pure-Python protobuf backend before any TensorFlow import, and by ensuring no early `google.protobuf` import can lock in the incompatible C++ implementation. Then I make the TFRecord ID/prediction alignment deterministic and correct by removing the non-deterministic dataset option and by deriving `test_ids` from the same base (non-augmented, non-repeated) dataset order as the TFRecords themselves. Finally, I keep your EfficientNetB0 + TTA inference core logic intact (same model, same transforms, same TTA count), while ensuring a valid `submission.csv` is always written in the exact `sample_submission.csv` row order.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "3"
os.environ["PROTOCOL_BUFFERS_PYTHON_DISABLE_CPP_IMPLEMENTATION"] = "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import re
import random
import math

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.backend as K

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
CANDIDATE_ROOTS = [
    "/kaggle/input/siim-isic-melanoma-classification",
    "/kaggle/data/siim-isic-melanoma-classification",
    "/kaggle/input",
    "/kaggle/data",
]

LOCAL_DS_PATH = None
for root in CANDIDATE_ROOTS:
    tfrec_dir = os.path.join(root, "tfrecords")
    if tf.io.gfile.exists(tfrec_dir):
        LOCAL_DS_PATH = root
        break

if LOCAL_DS_PATH is None:
    raise FileNotFoundError(
        "Could not locate dataset root containing a 'tfrecords' directory. "
        f"Tried: {CANDIDATE_ROOTS}"
    )

TFREC_DIR = os.path.join(LOCAL_DS_PATH, "tfrecords")

TEST_FILES = sorted(tf.io.gfile.glob(os.path.join(TFREC_DIR, "test*.tfrec")))
if len(TEST_FILES) == 0:
    raise FileNotFoundError(f"No TFRecord files found under: {TFREC_DIR}")

BATCH_SIZE = 256
print("Using dataset root:", LOCAL_DS_PATH)
print("Found test TFRecords:", len(TEST_FILES))



## === cell 2
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print("Running on TPU ", tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()


def count_data_items(filenames):
    n = [
        int(re.compile(r"-([0-9]*)\.").search(filename).group(1))
        for filename in filenames
    ]
    return int(np.sum(n))




## === cell 3
ROT_ = 180.0
SHR_ = 2.0
HZOOM_ = 8.0
WZOOM_ = 8.0
HSHIFT_ = 8.0
WSHIFT_ = 8.0


def get_mat(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
    rotation = math.pi * rotation / 180.0
    shear = math.pi * shear / 180.0

    def get_3x3_mat(lst):
        return tf.reshape(tf.concat([lst], axis=0), [3, 3])

    c1 = tf.math.cos(rotation)
    s1 = tf.math.sin(rotation)
    one = tf.constant([1], dtype="float32")
    zero = tf.constant([0], dtype="float32")

    rotation_matrix = get_3x3_mat([c1, s1, zero, -s1, c1, zero, zero, zero, one])
    c2 = tf.math.cos(shear)
    s2 = tf.math.sin(shear)

    shear_matrix = get_3x3_mat([one, s2, zero, zero, c2, zero, zero, zero, one])
    zoom_matrix = get_3x3_mat(
        [one / height_zoom, zero, zero, zero, one / width_zoom, zero, zero, zero, one]
    )
    shift_matrix = get_3x3_mat(
        [one, zero, height_shift, zero, one, width_shift, zero, zero, one]
    )

    return K.dot(K.dot(rotation_matrix, shear_matrix), K.dot(zoom_matrix, shift_matrix))


def transform(image, DIM=256):
    XDIM = DIM % 2  # fix for size 331

    rot = ROT_ * tf.random.normal([1], dtype="float32")
    shr = SHR_ * tf.random.normal([1], dtype="float32")
    h_zoom = 1.0 + tf.random.normal([1], dtype="float32") / HZOOM_
    w_zoom = 1.0 + tf.random.normal([1], dtype="float32") / WZOOM_
    h_shift = HSHIFT_ * tf.random.normal([1], dtype="float32")
    w_shift = WSHIFT_ * tf.random.normal([1], dtype="float32")

    m = get_mat(rot, shr, h_zoom, w_zoom, h_shift, w_shift)

    x = tf.repeat(tf.range(DIM // 2, -DIM // 2, -1), DIM)
    y = tf.tile(tf.range(-DIM // 2, DIM // 2), [DIM])
    z = tf.ones([DIM * DIM], dtype="int32")
    idx = tf.stack([x, y, z])

    idx2 = K.dot(m, tf.cast(idx, dtype="float32"))
    idx2 = K.cast(idx2, dtype="int32")
    idx2 = K.clip(idx2, -DIM // 2 + XDIM + 1, DIM // 2)

    idx3 = tf.stack([DIM // 2 - idx2[0,], DIM // 2 - 1 + idx2[1,]])
    d = tf.gather_nd(image, tf.transpose(idx3))

    return tf.reshape(d, [DIM, DIM, 3])


def data_augment_test(image, img_id):
    image = transform(image, DIM=128)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.rot90(image)
    return image, img_id


def process_test_data(data_file):
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string),
    }
    data = tf.io.parse_single_example(data_file, LABELED_TFREC_FORMAT)
    img = tf.image.decode_jpeg(data["image"], channels=3)
    img = tf.image.resize(img, (128, 128))
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.reshape(img, [128, 128, 3])
    img_id = data["image_name"]
    return img, img_id




## === cell 4
def efficientnetbx():
    backbone = tf.keras.applications.EfficientNetB0(
        input_shape=(128, 128, 3), include_top=False, weights=None
    )
    model = tf.keras.Sequential(
        [
            backbone,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    return model


def load_pretrained_backbone_weights(model):
    """
    Ensure we are not running with a randomly initialized backbone.
    We load pretrained EfficientNetB0 (ImageNet) and copy weights into our backbone.
    """
    pretrained = tf.keras.applications.EfficientNetB0(
        input_shape=(128, 128, 3), include_top=False, weights="imagenet"
    )
    model.layers[0].set_weights(pretrained.get_weights())




## === cell 5
TTA = 11

with strategy.scope():
    model = efficientnetbx()
    load_pretrained_backbone_weights(model)
    model.compile(optimizer="adam", loss="binary_crossentropy")

model.build((None, 128, 128, 3))

det_opts = tf.data.Options()
det_opts.experimental_deterministic = True

base_test_ds = (
    tf.data.TFRecordDataset(TEST_FILES, num_parallel_reads=1)
    .with_options(det_opts)
    .map(process_test_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
)

test_dataset = (
    base_test_ds.map(
        data_augment_test, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    .repeat()
    .batch(BATCH_SIZE * 4)
    .prefetch(tf.data.experimental.AUTOTUNE)
)

test_imgs = test_dataset.map(lambda images, ids: images)

ct_test = count_data_items(TEST_FILES)
STEPS = int(np.ceil(TTA * ct_test / (BATCH_SIZE * 4)))

pred = model.predict(test_imgs, steps=STEPS, verbose=1)[: TTA * ct_test, :]
predictions = np.mean(pred.reshape((ct_test, TTA), order="F"), axis=1)

test_ids = []
for _, batch_ids in base_test_ds.batch(1024).prefetch(tf.data.experimental.AUTOTUNE):
    test_ids.append(batch_ids.numpy().astype("U"))
test_ids = np.concatenate(test_ids, axis=0)
if test_ids.shape[0] != ct_test:
    ct_test = test_ids.shape[0]
    predictions = predictions[:ct_test]

sub = pd.DataFrame({"image_name": test_ids, "target": predictions})

sample_path = os.path.join(LOCAL_DS_PATH, "sample_submission.csv")
if not os.path.exists(sample_path):
    alt = "/kaggle/input/sample_submission.csv"
    if os.path.exists(alt):
        sample_path = alt
sample = pd.read_csv(sample_path)

sub = sample[["image_name"]].merge(sub, on="image_name", how="left")
sub["target"] = sub["target"].fillna(
    sub["target"].mean() if sub["target"].notna().any() else 0.5
)

sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print(
    "Target stats:",
    float(sub["target"].min()),
    float(sub["target"].mean()),
    float(sub["target"].max()),
)
