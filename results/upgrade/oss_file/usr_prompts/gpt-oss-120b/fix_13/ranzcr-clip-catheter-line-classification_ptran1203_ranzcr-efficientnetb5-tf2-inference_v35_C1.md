# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

# 4. Data file paths

```
/
    kaggle/
        data/
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 2 other files
                test/
                    1.2.826.0.1.3680043.8.498.11377666469497047083697986859684478212.jpg (162.5 kB)
                    1.2.826.0.1.3680043.8.498.51816564183045831083926719815892052332.jpg (322.2 kB)
                    ... and 3007 other files
                train/
                    1.2.826.0.1.3680043.8.498.86126240554015133345121504023675730690.jpg (136.3 kB)
                    1.2.826.0.1.3680043.8.498.85192731373652359757496590545802262751.jpg (189.5 kB)
                    ... and 27072 other files
        input/
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 2 other files
                test/
                    1.2.826.0.1.3680043.8.498.11377666469497047083697986859684478212.jpg (162.5 kB)
                    1.2.826.0.1.3680043.8.498.51816564183045831083926719815892052332.jpg (322.2 kB)
                    ... and 3007 other files
                train/
                    1.2.826.0.1.3680043.8.498.86126240554015133345121504023675730690.jpg (136.3 kB)
                    1.2.826.0.1.3680043.8.498.85192731373652359757496590545802262751.jpg (189.5 kB)
                    ... and 27072 other files
            test/
                test/
                    1.2.826.0.1.3680043.8.498.11377666469497047083697986859684478212.jpg (162.5 kB)
                    1.2.826.0.1.3680043.8.498.51816564183045831083926719815892052332.jpg (322.2 kB)
                    ... and 3007 other files
            train/
                train/
                    1.2.826.0.1.3680043.8.498.86126240554015133345121504023675730690.jpg (136.3 kB)
                    1.2.826.0.1.3680043.8.498.85192731373652359757496590545802262751.jpg (189.5 kB)
                    ... and 27072 other files
        working/
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 2 other files
                test/
                    1.2.826.0.1.3680043.8.498.11377666469497047083697986859684478212.jpg (162.5 kB)
                    1.2.826.0.1.3680043.8.498.51816564183045831083926719815892052332.jpg (322.2 kB)
                    ... and 3007 other files
                train/
                    1.2.826.0.1.3680043.8.498.86126240554015133345121504023675730690.jpg (136.3 kB)
                    1.2.826.0.1.3680043.8.498.85192731373652359757496590545802262751.jpg (189.5 kB)
                    ... and 27072 other files
```

-> data/ranzcr-clip-catheter-line-classification/sample_submission.csv has 3009 rows and 10 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline

-> data/ranzcr-clip-catheter-line-classification/train.csv has 27074 rows and 13 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline, CVC - Normal, Swan Ganz Catheter Present, PatientID

-> data/ranzcr-clip-catheter-line-classification/train_annotations.csv has 16261 rows and 3 columns.
The columns are: StudyInstanceUID, label, data

-> input/ranzcr-clip-catheter-line-classification/sample_submission.csv has 3009 rows and 10 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline

-> input/ranzcr-clip-catheter-line-classification/train.csv has 27074 rows and 13 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline, CVC - Normal, Swan Ganz Catheter Present, PatientID

-> input/ranzcr-clip-catheter-line-classification/train_annotations.csv has 16261 rows and 3 columns.
The columns are: StudyInstanceUID, label, data

-> working/ranzcr-clip-catheter-line-classification/sample_submission.csv has 3009 rows and 10 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf

tf_available = tf.__version__ is not None

W = H = 338
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
autotune = tf.data.experimental.AUTOTUNE if tf_available else None
BATCH_SIZE = 128  # used for both training and inference



## === cell 1
base_path = "../input/ranzcr-clip-catheter-line-classification"
sample_submission_path = os.path.join(base_path, "sample_submission.csv")
train_csv_path = os.path.join(base_path, "train.csv")
test_images_dir = os.path.join(base_path, "test")

submission_df = pd.read_csv(sample_submission_path, nrows=0)
target_cols = [c for c in submission_df.columns if c != "StudyInstanceUID"]
N_CLASSES = len(target_cols)

print("Target columns:", target_cols)
print("Number of classes:", N_CLASSES)



## === cell 2
features = {
    "StudyInstanceUID": tf.io.FixedLenFeature([], tf.string) if tf_available else None,
    "image": tf.io.FixedLenFeature([], tf.string) if tf_available else None,
}

if tf_available:

    def parse_example(sample):
        sample = tf.io.parse_single_example(sample, features)
        image = tf.image.decode_png(sample["image"], channels=3)
        image = tf.image.resize(image, (H, W))
        uid = sample["StudyInstanceUID"]
        return image, uid

    def preprocess(img, uid):
        img = tf.cast(img, tf.float32)
        img = (img - mean) / std
        return img, uid

else:
    parse_example = preprocess = None



## === cell 3
model = None
if tf_available:
    weight_dir = "../input/cassava2020weights"
    model_map = {
        "efficientb3": [
            tf.keras.applications.EfficientNetB3,
            os.path.join(weight_dir, "ranzcr_efficientb3.h5"),
        ],
        "efficientb5": [
            tf.keras.applications.EfficientNetB5,
            os.path.join(weight_dir, "ranzcr_efficientb5.h5"),
        ],
        "efficientb7": [
            tf.keras.applications.EfficientNetB7,
            os.path.join(weight_dir, "ranzcr_efficientb7.h5"),
        ],
        "resnet101": [
            tf.keras.applications.ResNet101,
            os.path.join(weight_dir, "ranzcr_resnet101.h5"),
        ],
    }

    def get_model(
        base_model,
        baseline_weight=None,
        init_weight=None,
        lr=0.001,
        optimizer=tf.optimizers.Adam,
    ):
        base = base_model(
            include_top=False,
            input_shape=(H, W, 3),
            pooling="avg",
            weights=baseline_weight,
        )
        x = tf.keras.layers.Dropout(0.3)(base.output)
        x = tf.keras.layers.Dense(N_CLASSES, activation="sigmoid")(x)
        m = tf.keras.models.Model(inputs=base.input, outputs=x)
        m.compile(
            optimizer=optimizer(learning_rate=lr),
            loss="binary_crossentropy",
            metrics=[tf.keras.metrics.AUC()],
        )
        if init_weight:
            try:
                m.load_weights(init_weight)
                print(f"Loaded pretrained weights from {init_weight}")
            except Exception as e:
                print(f"Failed to load weights from {init_weight}: {e}")
        return m

    try:
        base_mode, weight_path = model_map["efficientb3"]
        if os.path.exists(weight_path):
            model = get_model(base_mode, init_weight=weight_path)
        else:
            print("Pretrained weight file not found, using ImageNet weights only.")
            model = get_model(base_mode)  # ImageNet weights only
    except Exception as e:
        print("Error initializing model:", e)



## === cell 4
if tf_available and model is not None:
    print("Preparing training dataset...")
    train_df = pd.read_csv(train_csv_path)
    train_labels = train_df[target_cols].astype(np.float32).values
    train_image_paths = [
        os.path.join(base_path, "train", f"{uid}.jpg")
        for uid in train_df["StudyInstanceUID"]
    ]

    def _load_train(path, label):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img_bytes, channels=3)
        img = tf.image.resize(img, (H, W))
        img = tf.cast(img, tf.float32)
        img = (img - mean) / std
        return img, label

    train_ds = tf.data.Dataset.from_tensor_slices((train_image_paths, train_labels))
    train_ds = train_ds.map(_load_train, num_parallel_calls=autotune)
    train_ds = train_ds.shuffle(1024).batch(BATCH_SIZE).prefetch(autotune)

    model.fit(train_ds, epochs=2, verbose=1)



## === cell 5
if not tf_available or model is None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier

    def _extract_rgb_mean(paths):
        feats = []
        for p in paths:
            img = cv2.imread(p)
            if img is None:
                feats.append([0.0, 0.0, 0.0])
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mean_rgb = img.mean(axis=(0, 1)) / 255.0
            feats.append(mean_rgb.tolist())
        return np.array(feats, dtype=np.float32)

    train_df = pd.read_csv(train_csv_path)
    label_df = train_df[target_cols].astype(np.float32)

    train_image_paths = [
        os.path.join(base_path, "train", f"{uid}.jpg")
        for uid in train_df["StudyInstanceUID"]
    ]
    X_train = _extract_rgb_mean(train_image_paths)
    y_train = label_df.values

    clf = OneVsRestClassifier(LogisticRegression(max_iter=200, solver="lbfgs"))
    clf.fit(X_train, y_train)

    test_files = sorted(
        [
            f
            for f in os.listdir(test_images_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    )
    test_image_paths = [os.path.join(test_images_dir, f) for f in test_files]
    X_test = _extract_rgb_mean(test_image_paths)

    fallback_preds = clf.predict_proba(X_test).tolist()
    fallback_ids = [os.path.splitext(f)[0] for f in test_files]



## === cell 6
preds = []
image_ids = []

if tf_available and model is not None:
    test_tfrecords_dir = os.path.join(base_path, "test_tfrecords")
    if os.path.isdir(test_tfrecords_dir) and len(os.listdir(test_tfrecords_dir)) > 0:
        tfrecord_files = [
            os.path.join(test_tfrecords_dir, f) for f in os.listdir(test_tfrecords_dir)
        ]
        test_data = tf.data.TFRecordDataset(tfrecord_files)
        test_data = test_data.map(parse_example, num_parallel_calls=autotune)
        test_data = test_data.map(preprocess, num_parallel_calls=autotune)
    else:
        test_files = [
            os.path.join(test_images_dir, f)
            for f in sorted(os.listdir(test_images_dir))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        def _decode_test(path):
            img_bytes = tf.io.read_file(path)
            img = tf.image.decode_jpeg(img_bytes, channels=3)
            img = tf.image.resize(img, (H, W))
            img = tf.cast(img, tf.float32)
            uid = tf.strings.regex_replace(
                tf.strings.split(path, os.sep)[-1], r"\.(jpg|jpeg|png)$", ""
            )
            return img, uid

        test_ds = tf.data.Dataset.from_tensor_slices(test_files)
        test_data = test_ds.map(_decode_test, num_parallel_calls=autotune)
        test_data = test_data.map(
            lambda img, uid: ((img - mean) / std, uid), num_parallel_calls=autotune
        )

    img_ds = test_data.map(lambda img, uid: img)
    id_ds = test_data.map(lambda img, uid: uid)

    img_ds = img_ds.batch(BATCH_SIZE).prefetch(autotune)
    preds_array = model.predict(img_ds, verbose=0)
    preds = preds_array.tolist()

    for uid in id_ds.as_numpy_iterator():
        if isinstance(uid, bytes):
            uid = uid.decode()
        image_ids.append(uid)

else:
    preds = fallback_preds
    image_ids = fallback_ids

submission = pd.read_csv(sample_submission_path)
submission = submission.iloc[0:0]  # keep only header
submission["StudyInstanceUID"] = image_ids
submission[target_cols] = preds
submission.to_csv("submission.csv", index=False)
print("Submission written to submission.csv with shape:", submission.shape)
