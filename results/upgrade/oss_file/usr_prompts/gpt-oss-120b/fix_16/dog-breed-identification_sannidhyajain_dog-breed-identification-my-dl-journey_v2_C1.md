# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Given a dataset of images of dogs, predict the breed of each image.

## Metric
Multi Class Log Loss.

## Submission Format
For each image in the test set, you must predict a probability for each of the different breeds. The file should contain a header and have the following format:
```
id,affenpinscher,afghan_hound,..,yorkshire_terrier
000621fb3cbb32d8935728e48679680e,0.0083,0.0,...,0.0083
etc.
```

## Dataset Description
- `train.zip` - the training set, you are provided the breed for these dogs
- `test.zip` - the test set, you must predict the probability of each breed for each image
- `sample_submission.csv` - a sample submission file in the correct format
- `labels.csv` - the breeds for the images in the train set

# 2. Python version

3.13

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 1 other files
                test/
                    a6d7c6cc8162c58d6f75d6f46cd6e0d4.jpg (39.4 kB)
                    280d29e9e2f9cf53e8deba4dc1d352fb.jpg (50.2 kB)
                    ... and 1021 other files
                train/
                    f37af9dee180ed1195985decf4ef7111.jpg (18.4 kB)
                    547e9cca520970b35c1d9acb0a0fb902.jpg (32.5 kB)
                    ... and 9197 other files
        input/
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 1 other files
                test/
                    a6d7c6cc8162c58d6f75d6f46cd6e0d4.jpg (39.4 kB)
                    280d29e9e2f9cf53e8deba4dc1d352fb.jpg (50.2 kB)
                    ... and 1021 other files
                train/
                    f37af9dee180ed1195985decf4ef7111.jpg (18.4 kB)
                    547e9cca520970b35c1d9acb0a0fb902.jpg (32.5 kB)
                    ... and 9197 other files
            test/
                test/
                    a6d7c6cc8162c58d6f75d6f46cd6e0d4.jpg (39.4 kB)
                    280d29e9e2f9cf53e8deba4dc1d352fb.jpg (50.2 kB)
                    ... and 1021 other files
            train/
                train/
                    f37af9dee180ed1195985decf4ef7111.jpg (18.4 kB)
                    547e9cca520970b35c1d9acb0a0fb902.jpg (32.5 kB)
                    ... and 9197 other files
        working/
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 1 other files
                test/
                    a6d7c6cc8162c58d6f75d6f46cd6e0d4.jpg (39.4 kB)
                    280d29e9e2f9cf53e8deba4dc1d352fb.jpg (50.2 kB)
                    ... and 1021 other files
                train/
                    f37af9dee180ed1195985decf4ef7111.jpg (18.4 kB)
                    547e9cca520970b35c1d9acb0a0fb902.jpg (32.5 kB)
                    ... and 9197 other files
```

-> data/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> input/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> input/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> working/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> working/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

# 5. Code solution

## === cell 0
import os, sys, numpy as np, pandas as pd

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import log_loss
from sklearn.decomposition import PCA
from PIL import Image

TF_AVAILABLE = False
print("TensorFlow import skipped; using sklearn fallback.")



## === cell 1
print("TensorFlow available:", TF_AVAILABLE)



## === cell 2
labels_path = "/kaggle/input/dog-breed-identification/labels.csv"
labels = pd.read_csv(labels_path)
print("Loaded", len(labels), "labels")



## === cell 3
unique_breeds = np.sort(labels["breed"].unique())
breed_to_idx = {b: i for i, b in enumerate(unique_breeds)}
idx_to_breed = {i: b for b, i in breed_to_idx.items()}
num_classes = len(unique_breeds)
print("Number of breeds:", num_classes)



## === cell 4
train_dir = "/kaggle/input/dog-breed-identification/train/"
test_dir = "/kaggle/input/dog-breed-identification/test/"

train_files = [os.path.join(train_dir, f"{img_id}.jpg") for img_id in labels["id"]]
train_labels_idx = labels["breed"].map(breed_to_idx).values

IMG_SIZE = 64  # modest size to keep memory low


def load_and_preprocess(paths):
    """Load JPEG files, resize to IMG_SIZE×IMG_SIZE, normalize to [0,1],
    and flatten to 1‑D vectors."""
    arr = np.empty((len(paths), IMG_SIZE * IMG_SIZE * 3), dtype=np.float32)
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            im = im.convert("RGB")
            im = im.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            data = np.asarray(im, dtype=np.float32) / 255.0
            arr[i] = data.ravel()
    return arr


X_full = load_and_preprocess(train_files)

X_train_raw, X_val_raw, y_train, y_val, train_files_split, val_files_split = (
    train_test_split(
        X_full,
        train_labels_idx,
        train_files,
        test_size=0.1,
        random_state=42,
        stratify=train_labels_idx,
    )
)

max_components = min(X_train_raw.shape[0], X_train_raw.shape[1])
PCA_COMPONENTS = min(200, max_components)  # 200 or less, never exceeds limits
pca = PCA(
    n_components=PCA_COMPONENTS,
    svd_solver="randomized",
    random_state=42,
)

pca.fit(X_train_raw)

X_train = pca.transform(X_train_raw)
X_val = pca.transform(X_val_raw)
X_full = pca.transform(X_full)  # reuse later for the full‑data model



## === cell 5
logreg = LogisticRegression(
    multi_class="multinomial",
    solver="saga",
    max_iter=10000,
    C=10.0,
    n_jobs=-1,
    verbose=0,
)
logreg.fit(X_train, y_train)
val_pred_lr = logreg.predict_proba(X_val)
val_logloss_lr = log_loss(y_val, val_pred_lr)
print(f"Validation log‑loss (LogisticRegression): {val_logloss_lr:.4f}")

knn = KNeighborsClassifier(
    n_neighbors=5,
    weights="distance",
    n_jobs=-1,
)
knn.fit(X_train, y_train)
val_pred_knn = knn.predict_proba(X_val)
val_logloss_knn = log_loss(y_val, val_pred_knn)
print(f"Validation log‑loss (KNN): {val_logloss_knn:.4f}")

if val_logloss_lr <= val_logloss_knn:
    best_model_name = "LogisticRegression"
    best_model = logreg
    best_val_logloss = val_logloss_lr
else:
    best_model_name = "KNN"
    best_model = knn
    best_val_logloss = val_logloss_knn
print(
    f"Chosen model: {best_model_name} with validation log‑loss {best_val_logloss:.4f}"
)

best_model.fit(X_full, train_labels_idx)

test_files = sorted(
    [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(".jpg")]
)
X_test = load_and_preprocess(test_files)
X_test = pca.transform(X_test)
test_preds = best_model.predict_proba(X_test)  # (num_test, num_classes)



## === cell 6
sample_sub_path = "/kaggle/input/dog-breed-identification/sample_submission.csv"
sample_sub = pd.read_csv(sample_sub_path, nrows=1)
breed_columns = sample_sub.columns.tolist()[1:]  # skip 'id'

pred_df = pd.DataFrame(
    test_preds, columns=[idx_to_breed[i] for i in range(num_classes)]
)
pred_df = pred_df[breed_columns]  # reorder columns to match submission format
pred_df.insert(0, "id", [os.path.basename(p)[:-4] for p in test_files])



## === cell 7
submission_path = "submission.csv"
pred_df.to_csv(submission_path, index=False)
print("Submission written to:", submission_path)



## === cell 8
print("File size (bytes):", os.path.getsize(submission_path))
print(pred_df.head())
