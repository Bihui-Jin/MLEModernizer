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

# 5. Target score

0.7669549828163248

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.49197) has done: 'I fix the TensorFlow import failure handling, derive the target columns directly from the sample‑submission header, adjust the class count accordingly, and correct the model prediction call (remove the erroneous `.numpy()`). Dummy predictions now match the actual number of columns, and the submission CSV be built with the proper columns.'
- What this solution (achieved 0.46256) has done: 'Implemented batch inference for the test set to eliminate per‑image prediction overhead. The test dataset is now batched, and predictions are obtained with a single `model.predict_on_batch` call per batch, preserving the original ordering and prediction logic. This reduces I/O and TensorFlow call overhead, keeping the core model unchanged while fitting comfortably within the 600‑second limit.'
- What this solution (achieved 0.5) has done: 'We replace the Python‑cv2 image loading loops with a fully TensorFlow data pipeline that reads, decodes and resizes JPEG files directly on the GPU/CPU in parallel. This removes the per‑image Python overhead during training and inference while keeping exactly the same preprocessing (RGB conversion, resize, mean‑std normalization). The model architecture, loss, optimizer, number of epochs and batch size stay unchanged, so results remain identical but the runtime drops well below the 600 s limit.'

# 9. Code solution

## === cell 0
import os, sys, numpy as np, pandas as pd, cv2, matplotlib.pyplot as plt
from datetime import datetime as dt
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

try:
    import tensorflow as tf

    tf_available = True
except Exception as e:
    print("TensorFlow import failed:", e)
    tf_available = False




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
W = H = 338
autotune = tf.data.experimental.AUTOTUNE if tf_available else None

features = {
    "StudyInstanceUID": tf.io.FixedLenFeature([], tf.string) if tf_available else None,
    "image": tf.io.FixedLenFeature([], tf.string) if tf_available else None,
}

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]




## === cell 2
base_path = "../input/ranzcr-clip-catheter-line-classification"
sample_submission_path = os.path.join(base_path, "sample_submission.csv")
train_csv_path = os.path.join(base_path, "train.csv")
test_images_dir = os.path.join(base_path, "test")

submission_df = pd.read_csv(sample_submission_path, nrows=0)
print("Submission columns:", submission_df.columns.tolist())

target_cols = [c for c in submission_df.columns if c != "StudyInstanceUID"]
N_CLASSES = len(target_cols)  # adjust class count to match submission format




## === cell 3
if tf_available:

    def parse_example(sample):
        sample = tf.io.parse_single_example(sample, features)
        image = tf.image.decode_png(sample["image"], channels=3)
        image = tf.image.resize(image, (H, W))
        image_id = sample["StudyInstanceUID"]
        return image, image_id

    def preprocess(images, labels):
        images = tf.cast(images, tf.float32)
        return (images - mean) / std, labels

    def get_count(fname):
        return 1868 if "15-1881.tfrec" in fname else 1881

    def get_tfrecord(indices, train_tfrecords, test_tfrecords_dir):
        files = [os.path.join(test_tfrecords_dir, train_tfrecords[i]) for i in indices]
        sizes = [get_count(file) for file in files]
        return files, sum(sizes)

    def get_model(
        base_model,
        baseline_weight=None,
        init_weight=None,
        lr=0.001,
        optimizer=tf.optimizers.Adam,
    ):
        base_model = base_model(
            include_top=False,
            input_shape=(W, H, 3),
            pooling="avg",
            weights=baseline_weight,
        )
        out = tf.keras.layers.Dropout(0.3)(base_model.output)
        out = tf.keras.layers.Dense(N_CLASSES, activation="sigmoid")(out)
        model = tf.keras.models.Model(inputs=base_model.input, outputs=out)
        model.compile(
            optimizer=optimizer(learning_rate=lr),
            loss="binary_crossentropy",
            metrics=[tf.keras.metrics.AUC()],
        )
        if init_weight:
            try:
                model.load_weights(init_weight)
                print(f"Weight loaded from {init_weight}")
            except Exception as e:
                print(f"Load weight from {init_weight} failed, {e}")
        return model

else:
    parse_example = preprocess = get_count = get_tfrecord = get_model = None




## === cell 4
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
    try:
        base_mode, weight_path = model_map["efficientb3"]
        if os.path.exists(weight_path):
            model = get_model(base_mode, init_weight=weight_path)
        else:
            print("Pretrained weight file not found at:", weight_path)
            model = get_model(base_mode)  # initialise with ImageNet weights only
    except Exception as e:
        print("Error loading model:", e)




## === cell 5
if not tf_available or model is None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier

    def _extract_rgb_mean(paths):
        """Return an (n_samples, 3) array of normalized RGB channel means."""
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

    SAMPLE_SIZE = len(train_df)
    train_subset = train_df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(
        drop=True
    )
    label_subset = label_df.loc[train_subset.index].values

    train_image_paths = [
        os.path.join(base_path, "train", f"{uid}.jpg")
        for uid in train_subset["StudyInstanceUID"]
    ]

    X_train = _extract_rgb_mean(train_image_paths)
    y_train = label_subset

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
BATCH_SIZE = 128  # larger batch to reduce iteration overhead

preds = []
image_ids = []

if tf_available and model is not None:
    test_tfrecords_dir = os.path.join(base_path, "test_tfrecords")
    if os.path.isdir(test_tfrecords_dir) and len(os.listdir(test_tfrecords_dir)) > 0:
        test_tcords = os.listdir(test_tfrecords_dir)
        files = [os.path.join(test_tfrecords_dir, c) for c in test_tcords]
        test_data = tf.data.TFRecordDataset(files)
        test_data = test_data.map(parse_example, num_parallel_calls=autotune)
        test_data = test_data.map(preprocess)
    else:
        test_files = [
            os.path.join(test_images_dir, fname)
            for fname in sorted(os.listdir(test_images_dir))
            if fname.lower().endswith((".jpg", ".jpeg", ".png"))
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
        test_data = test_data.map(lambda img, uid: ((img - mean) / std, uid))

    image_dataset = test_data.map(lambda img, uid: img)
    id_dataset = test_data.map(lambda img, uid: uid)

    image_dataset = image_dataset.batch(BATCH_SIZE).prefetch(autotune)

    preds_array = model.predict(image_dataset, verbose=0)
    preds = preds_array.tolist()

    image_ids = [
        uid.numpy().decode() for uid in id_dataset.unbatch().as_numpy_iterator()
    ]

else:
    if "fallback_preds" in globals():
        preds = fallback_preds
        image_ids = fallback_ids
    else:

        def dummy_prediction():
            return [0.0] * N_CLASSES

        for fname in os.listdir(test_images_dir):
            uid = os.path.splitext(fname)[0]
            image_ids.append(uid)
            preds.append(dummy_prediction())

submission = pd.read_csv(sample_submission_path)
submission = submission.iloc[0:0]  # keep only header
submission["StudyInstanceUID"] = image_ids
submission[target_cols] = preds
submission.to_csv("submission.csv", index=False)
print("Submission written to submission.csv with shape:", submission.shape)

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_13/3146948606.py in <cell line: 0>()
     43 
     44     image_ids = [
---> 45         uid.numpy().decode() for uid in id_dataset.unbatch().as_numpy_iterator()
     46     ]
     47 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in unbatch(self, name)
   2981     # pylint: disable=g-import-not-at-top,protected-access
   2982     from tensorflow.python.data.ops import unbatch_op
-> 2983     return unbatch_op._unbatch(self, name=name)
   2984     # pylint: enable=g-import-not-at-top,protected-access
   2985 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/unbatch_op.py in _unbatch(input_dataset, name)
     24   """See `Dataset.unbatch()` for details."""
     25   normalized_dataset = dataset_ops.normalize_to_dense(input_dataset)
---> 26   return _UnbatchDataset(normalized_dataset, name=name)
     27 
     28 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/unbatch_op.py in __init__(self, input_dataset, name)
     34     flat_shapes = input_dataset._flat_shapes  # pylint: disable=protected-access
     35     if any(s.ndims == 0 for s in flat_shapes):
---> 36       raise ValueError("Cannot unbatch an input with scalar components.")
     37     known_batch_dim = tensor_shape.Dimension(None)
     38     for s in flat_shapes:

ValueError: Cannot unbatch an input with scalar components.
