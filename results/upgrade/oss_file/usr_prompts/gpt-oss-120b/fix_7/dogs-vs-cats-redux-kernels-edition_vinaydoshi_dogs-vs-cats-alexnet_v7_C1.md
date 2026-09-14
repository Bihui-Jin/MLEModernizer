# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Given a dataset of images of dogs and cats, predict if an image is a dog or a cat.

## Metric
Log loss.

## Submission Format
For each image in the test set, you must submit a probability that image is a dog. The file should have a header and be in the following format:

```
id,label
1,0.5
2,0.5
3,0.5
...
```

## Dataset
The train folder contains 25,000 images of dogs and cats. Each image in this folder has the label as part of the filename. The test folder contains 12,500 images, named according to a numeric id.

# 2. Python version

3.8

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
        input/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
        working/
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
```

-> data/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> working/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

# 5. Code solution

## === cell 0
import os, numpy as np, pandas as pd, cv2, h5py, tqdm, json, csv, multiprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.ensemble import GradientBoostingClassifier  # new model




## === cell 1
image_types = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def list_images(basePath, contains=None):
    for root, _, files in os.walk(basePath):
        for filename in files:
            if contains and contains not in filename:
                continue
            ext = os.path.splitext(filename)[1].lower()
            if ext in image_types:
                yield os.path.join(root, filename)


def extract_features(image_path, size=(32, 32)):
    """Read an image once, resize, and return concatenated mean RGB + flattened RGB."""
    img = cv2.imread(image_path)
    if img is None:
        return np.zeros(3 + size[0] * size[1] * 3, dtype=np.float32)
    img = cv2.resize(img, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mean_rgb = img.mean(axis=(0, 1))
    flat_rgb = img.flatten()
    return np.concatenate([mean_rgb, flat_rgb]).astype(np.float32)




## === cell 2
train_path = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/train/"
test_path = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/test/"

train_img_paths = list(list_images(train_path))
test_img_paths = list(list_images(test_path))

print(
    f"Found {len(train_img_paths)} training images and {len(test_img_paths)} test images."
)




## === cell 3
train_labels = []
for p in train_img_paths:
    name = os.path.basename(p)
    label_str = name.split(".")[0]  # 'cat' or 'dog'
    train_labels.append(label_str)

le = LabelEncoder()
train_labels_enc = le.fit_transform(train_labels)  # cat=0, dog=1

if len(train_img_paths) == 0:
    print("No training images found – skipping model training.")
    train_imgs, val_imgs, y_train, y_val = [], [], [], []
else:
    train_imgs, val_imgs, y_train, y_val = train_test_split(
        train_img_paths,
        train_labels_enc,
        test_size=0.2,
        random_state=42,
        stratify=train_labels_enc,
    )




## === cell 4
def build_feature_matrix(paths):
    """Efficiently build a (n_samples, n_features) matrix using parallel extraction."""
    if not paths:
        return np.empty((0, 3 + 32 * 32 * 3), dtype=np.float32)
    n_samples = len(paths)
    n_features = 3 + 32 * 32 * 3
    feats = np.empty((n_samples, n_features), dtype=np.float32)

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        for i, feature in enumerate(
            tqdm.tqdm(
                pool.imap(extract_features, paths),
                total=n_samples,
                desc="Extracting features",
            )
        ):
            feats[i] = feature
    return feats


if len(train_imgs) == 0:
    sample_sub_path = (
        "/kaggle/input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv"
    )
    sample_sub = pd.read_csv(sample_sub_path)
    sample_sub["label"] = 0.5
    submission_path = "/kaggle/working/submission.csv"
    sample_sub.to_csv(submission_path, index=False)
    print(f"Submission (fallback constant) written to {submission_path}")
else:
    X_train = build_feature_matrix(train_imgs)
    X_val = build_feature_matrix(val_imgs)

    gbc = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
    )
    gbc.fit(X_train, y_train)

    val_pred = gbc.predict_proba(X_val)[:, 1]
    val_loss = log_loss(y_val, val_pred)
    print(f"Validation log loss: {val_loss:.5f}")

    X_test = build_feature_matrix(test_img_paths)
    test_pred = gbc.predict_proba(X_test)[:, 1]

    sample_sub_path = (
        "/kaggle/input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv"
    )
    sample_sub = pd.read_csv(sample_sub_path)

    id_to_prob = {}
    for path, prob in zip(test_img_paths, test_pred):
        fname = os.path.basename(path)
        id_str = os.path.splitext(fname)[0]  # e.g., "1234"
        if "." in id_str:
            id_str = id_str.split(".")[-1]
        id_to_prob[int(id_str)] = prob

    sample_sub["label"] = sample_sub["id"].map(id_to_prob).fillna(0.5)

    submission_path = "/kaggle/working/submission.csv"
    sample_sub.to_csv(submission_path, index=False)
    print(f"Submission written to {submission_path}")




## === cell 5
if os.path.exists(submission_path):
    sub_check = pd.read_csv(submission_path)
    print("Submission preview:")
    print(sub_check.head())
    print(f"Rows: {len(sub_check)}")
else:
    print("Error: submission file not created.")
