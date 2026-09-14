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

0.8848594741613781

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I set the protobuf implementation flag before importing TensorFlow to avoid the import error, then safely handle the missing model file with a fallback that predicts the most‑common class from the training data. Finally, I generate the submission directly from the sample submission file using this majority‑class prediction, ensuring a correctly formatted `.csv` file is written.'
- What this solution (achieved 0.61099) has done: 'I wrapped the TensorFlow import in a safe try/except and provided fall‑back constants so the script no longer crashes when TF cannot be loaded. If TensorFlow is unavailable, I train a very lightweight LogisticRegression model on down‑scaled image pixels using scikit‑learn (which is present in the environment) to replace the naïve majority‑class dummy. The model is trained on a subset of the training images for speed, then used to predict the test set and write a correctly formatted `submission.csv`.'
- What this solution (achieved 0.43386) has done: 'I replace the dummy‑zero input with the actual test images, loading each image, resizing it to the expected size, and feeding the real data through the fallback model. This keeps the original fallback logic but provides meaningful features, which should improve the classification accuracy toward the target score while preserving the rest of the pipeline.'
- What this solution (achieved 0.4204) has done: 'The update keeps the original workflow but replaces the slow `saga` solver with the much faster `lbfgs` solver (still a multinomial logistic regression) and caps its iteration count at 200 to stay well within the time limit while preserving the same model type and feature handling. No changes are made to data loading, preprocessing, or prediction logic, so the resulting predictions remain equivalent to the original approach.'
- What this solution (achieved 0.4204) has done: 'I wrap the TensorFlow import in a safe try/except block so that if the protobuf incompatibility triggers an error the script continues using the sklearn fallback model. This prevents the AttributeError on import and still produces a correctly formatted `submission.csv` while keeping the core logic unchanged.'

# 9. Code solution

## === cell 0
AUTO = None
IMAGE_SIZE = (64, 64)  # reduced size to limit memory usage while preserving content
BATCH_SIZE_PER_REPLICA = 8
NUM_CLASSES = 5
BATCH_SIZE = 8


## === cell 1
train_csv_path = os.path.join(DATA_DIR, "train.csv")
train_df = pd.read_csv(train_csv_path)

if model is None:
    from pathlib import Path
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.neural_network import MLPClassifier

    train_images_dir = Path(DATA_DIR) / "train_images"

    train_items = [
        (str(train_images_dir / img_id), int(label))
        for img_id, label in zip(train_df["image_id"], train_df["label"])
    ]

    n_train = len(train_items)
    feat_len = IMAGE_SIZE[0] * IMAGE_SIZE[1] * 3
    X_train = np.empty((n_train, feat_len), dtype=np.float32)
    y_train = np.empty(n_train, dtype=np.int32)

    def load_train_item(item):
        path, label = item
        try:
            img = Image.open(path).convert("RGB")
            img = img.resize(IMAGE_SIZE)
            arr = np.asarray(img, dtype=np.float32).ravel()
        except Exception:
            arr = np.zeros(feat_len, dtype=np.float32)
        return arr, label

    cpu_cnt = multiprocessing.cpu_count()
    max_workers = min(cpu_cnt, 8)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, (arr, label) in enumerate(executor.map(load_train_item, train_items)):
            X_train[idx] = arr
            y_train[idx] = label

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    pca = PCA(n_components=200, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)

    clf = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        max_iter=30,
        batch_size=256,
        random_state=42,
        verbose=False,
    )
    clf.fit(X_train_pca, y_train)

    class SklearnFallbackModel:
        def __init__(self, classifier, scaler, pca):
            self.clf = classifier
            self.scaler = scaler
            self.pca = pca

        def predict(self, x, verbose=0):
            batch = x.shape[0]
            flat = x.reshape(batch, -1).astype(np.float32)
            flat = self.scaler.transform(flat)
            flat = self.pca.transform(flat)
            probs = self.clf.predict_proba(flat)
            return probs

    model = SklearnFallbackModel(clf, scaler, pca)


## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2371776631.py in <cell line: 0>()
----> 1 train_csv_path = os.path.join(DATA_DIR, "train.csv")
      2 train_df = pd.read_csv(train_csv_path)
      3 
      4 if model is None:
      5     from pathlib import Path

NameError: name 'os' is not defined

## === cell 2
sample_sub_path = os.path.join(DATA_DIR, "sample_submission.csv")
sample_sub_df = pd.read_csv(sample_sub_path)

test_images_dir = os.path.join(DATA_DIR, "test_images")
test_items = [
    os.path.join(test_images_dir, img_id) for img_id in sample_sub_df["image_id"]
]


def load_test_image(path):
    try:
        img = Image.open(path).convert("RGB")
        img = img.resize(IMAGE_SIZE)
        return np.asarray(img, dtype=np.float32)
    except Exception:
        return np.zeros((*IMAGE_SIZE, 3), dtype=np.float32)


cpu_cnt = multiprocessing.cpu_count()
max_workers = min(cpu_cnt, 8)

with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    test_imgs = list(executor.map(load_test_image, test_items))

X_test = np.stack(test_imgs, axis=0)

pred_probs = model.predict(X_test, verbose=0)
pred_labels = pred_probs.argmax(axis=1).astype(int)

prediction_df = pd.DataFrame(
    {
        "image_id": sample_sub_df["image_id"],
        "label": pred_labels,
    }
)

prediction_df["label"] = prediction_df["label"].astype(int)

submission_path = "submission.csv"
prediction_df.to_csv(submission_path, index=False)

print(f"Submission file created at {submission_path} with {len(prediction_df)} rows.")

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2036426487.py in <cell line: 0>()
----> 1 sample_sub_path = os.path.join(DATA_DIR, "sample_submission.csv")
      2 sample_sub_df = pd.read_csv(sample_sub_path)
      3 
      4 test_images_dir = os.path.join(DATA_DIR, "test_images")
      5 test_items = [

NameError: name 'os' is not defined
