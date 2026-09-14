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

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import cv2

try:
    import tensorflow as tf
except Exception:
    tf = None

W = H = 456
N_CLASSES = 11
target_cols = [
    "CVC - Abnormal",
    "CVC - Borderline",  # fixed spelling
    "CVC - Normal",
    "ETT - Abnormal",
    "ETT - Borderline",
    "ETT - Normal",
    "NGT - Abnormal",
    "NGT - Borderline",
    "NGT - Incompletely Imaged",
    "NGT - Normal",
    "Swan Ganz Catheter Present",
]



## === cell 1
if tf is not None:
    autotune = tf.data.experimental.AUTOTUNE
    features = {
        "StudyInstanceUID": tf.io.FixedLenFeature([], tf.string),
        "image": tf.io.FixedLenFeature([], tf.string),
    }

    def parse_example(sample):
        sample = tf.io.parse_single_example(sample, features)
        image = tf.image.decode_png(sample["image"], channels=3)
        image = tf.image.resize(image, (H, W))
        image_id = sample["StudyInstanceUID"]
        return image, image_id

    def preprocess(images, labels):
        images = tf.cast(images, tf.float32)
        return (images - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225], labels

else:
    parse_example = None
    preprocess = None



## === cell 2
if tf is not None:
    test_tfrecords_dir = (
        "../input/ranzcr-clip-catheter-line-classification/test_tfrecords"
    )
    if os.path.isdir(test_tfrecords_dir):
        test_tfrecords = os.listdir(test_tfrecords_dir)
        files = [os.path.join(test_tfrecords_dir, c) for c in test_tfrecords]
        test_data = tf.data.TFRecordDataset(files)
        test_data = test_data.map(parse_example, num_parallel_calls=autotune)
        test_data = test_data.map(preprocess)
        test_data = test_data.prefetch(1)
    else:
        test_data = []
else:
    test_data = []  # placeholder when TensorFlow is unavailable




## === cell 3
def show_samples(dataset):
    if tf is None:
        print("TensorFlow not available – skipping visualisation.")
        return
    rows = cols = 2
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15, 15))
    for i, (img, _) in enumerate(dataset.shuffle(100).take(rows * cols)):
        fig.add_subplot(rows, cols, i + 1)
        plt.imshow(img / 255)
    plt.show()


if isinstance(test_data, list):
    print("Test data is not a tf.data.Dataset – skipping visualisation.")
else:
    show_samples(test_data)



## === cell 4
if tf is not None:
    weight_dir = "../input/cassava2020weights"
    model_map = {
        "efficientb5": [
            tf.keras.applications.EfficientNetB5,
            os.path.join(weight_dir, "ranzcr_efficientb5.h5"),
        ]
    }
    base_mode, weight_path = model_map["efficientb5"]

    def get_model(base_model, init_weight=None, lr=0.001, optimizer=tf.optimizers.Adam):
        base = base_model(
            include_top=False, input_shape=(W, H, 3), pooling="avg", weights=None
        )
        x = tf.keras.layers.Dropout(0.3)(base.output)
        out = tf.keras.layers.Dense(N_CLASSES, activation="sigmoid")(x)
        model = tf.keras.models.Model(inputs=base.input, outputs=out)
        model.compile(
            optimizer=optimizer(learning_rate=lr),
            loss="binary_crossentropy",
            metrics=[tf.keras.metrics.AUC()],
        )
        if init_weight and os.path.exists(init_weight):
            try:
                model.load_weights(init_weight)
                print(f"Weight loaded from {init_weight}")
            except Exception as e:
                print(f"Failed to load weights: {e}")
        return model

    model = get_model(base_mode, init_weight=weight_path)
else:
    model = None
    print("Model not created – TensorFlow unavailable.")



## === cell 5
sample_sub_path = (
    "../input/ranzcr-clip-catheter-line-classification/sample_submission.csv"
)
if not os.path.exists(sample_sub_path):
    sample_sub_path = "../input/sample_submission.csv"  # fallback
submission_df = pd.read_csv(sample_sub_path)

for col in target_cols:
    if col not in submission_df.columns:
        submission_df[col] = np.nan

train_path = "../input/ranzcr-clip-catheter-line-classification/train.csv"
if not os.path.exists(train_path):
    train_path = "../input/train.csv"  # fallback
train_df = pd.read_csv(train_path)

prevalence = train_df[target_cols].mean().values.astype(np.float32)

if model is not None and isinstance(test_data, (list, tuple)) and len(test_data) > 0:
    preds = []
    for image, img_id in test_data:
        tta_imgs = tf.stack([image, tf.image.flip_left_right(image)])
        pred = model.predict_on_batch(tta_imgs).tolist()
        pred = np.max(pred, axis=0)
        preds.append(pred)
    preds = np.array(preds)
    submission_df = submission_df.set_index("StudyInstanceUID")
    pred_df = pd.DataFrame(
        preds,
        index=[img_id.numpy().decode() for _, img_id in test_data],
        columns=target_cols,
    )
    fill_dict = dict(zip(target_cols, prevalence))
    pred_df = pred_df.reindex(submission_df.index).fillna(fill_dict)
    submission_df.loc[:, target_cols] = pred_df.values
else:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import IncrementalPCA

    base_dir = os.path.dirname(train_path)
    train_img_dir = os.path.join(base_dir, "train")
    test_img_dir = os.path.join(os.path.dirname(sample_sub_path), "test")

    def extract_features(img_resized):
        flat = img_resized.ravel().astype(np.float32)

        channel_means = img_resized.mean(axis=(0, 1)).astype(np.float32)  # (3,)
        channel_stds = img_resized.std(axis=(0, 1)).astype(np.float32)  # (3,)

        hist_features = []
        for ch in range(3):
            hist = cv2.calcHist([img_resized], [ch], None, [16], [0, 256]).flatten()
            if hist.sum() != 0:
                hist = hist / hist.sum()
            hist_features.append(hist.astype(np.float32))
        hist_features = np.concatenate(hist_features)  # (48,)

        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200).flatten().astype(np.float32) / 255.0

        return np.concatenate([flat, channel_means, channel_stds, hist_features, edges])

    sample_df = train_df

    X_train_list, y_train_list = [], []
    for _, row in sample_df.iterrows():
        img_path = os.path.join(train_img_dir, f"{row['StudyInstanceUID']}.jpg")
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_resized = cv2.resize(img, (128, 128))
        X_train_list.append(extract_features(img_resized))
        y_train_list.append(row[target_cols].values.astype(np.float32))

    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.float32)

    if X_train.size > 0:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)

        n_components = min(500, X_scaled.shape[1])
        ipca = IncrementalPCA(n_components=n_components, batch_size=1024)
        X_pca = ipca.fit_transform(X_scaled)

        classifiers = []
        for i in range(N_CLASSES):
            clf = LogisticRegression(
                max_iter=5000, class_weight="balanced", solver="lbfgs"
            )
            clf.fit(X_pca, y_train[:, i])
            classifiers.append(clf)

        test_ids = submission_df["StudyInstanceUID"].values
        X_test_list, X_test_flip_list, valid_ids = [], [], []
        for uid in test_ids:
            img_path = os.path.join(test_img_dir, f"{uid}.jpg")
            if not os.path.exists(img_path):
                continue
            img = cv2.imread(img_path)
            if img is None:
                continue
            img_resized = cv2.resize(img, (128, 128))
            X_test_list.append(extract_features(img_resized))
            img_flipped = cv2.flip(img_resized, 1)
            X_test_flip_list.append(extract_features(img_flipped))
            valid_ids.append(uid)

        X_test = np.array(X_test_list, dtype=np.float32)
        X_test_flip = np.array(X_test_flip_list, dtype=np.float32)

        if X_test.size > 0:
            X_test_scaled = scaler.transform(X_test)
            X_test_flip_scaled = scaler.transform(X_test_flip)

            X_test_pca = ipca.transform(X_test_scaled)
            X_test_flip_pca = ipca.transform(X_test_flip_scaled)

            preds_orig = np.column_stack(
                [clf.predict_proba(X_test_pca)[:, 1] for clf in classifiers]
            )
            preds_flip = np.column_stack(
                [clf.predict_proba(X_test_flip_pca)[:, 1] for clf in classifiers]
            )
            preds = (preds_orig + preds_flip) / 2.0

            pred_df = pd.DataFrame(preds, index=valid_ids, columns=target_cols)

            fill_dict = dict(zip(target_cols, prevalence))
            pred_df = pred_df.reindex(submission_df["StudyInstanceUID"]).fillna(
                fill_dict
            )

            submission_df.loc[:, target_cols] = pred_df.values
        else:
            print("No test images found – using prevalence as fallback.")
            submission_df[target_cols] = prevalence
    else:
        print("No training images processed – using prevalence as fallback.")
        submission_df[target_cols] = prevalence

submission_df.to_csv("submission.csv", index=False)
print("Submission file 'submission.csv' written with shape:", submission_df.shape)
