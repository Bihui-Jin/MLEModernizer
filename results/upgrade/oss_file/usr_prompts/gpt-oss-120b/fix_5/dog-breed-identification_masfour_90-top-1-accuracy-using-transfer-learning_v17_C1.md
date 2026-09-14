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

3.8

# 3. Installed packages

geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
seaborn==0.12.2
sklearn-pandas==2.2.0
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
        input/
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
        working/
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
```

-> data/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> data/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> input/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> input/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os, shutil
import numpy as np, pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from tqdm import tqdm



## === cell 1
BASE_DIR = "../input/dog-breed-identification"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
SAMPLE_SUBMIT = os.path.join(BASE_DIR, "sample_submission.csv")
LABELS_CSV = os.path.join(BASE_DIR, "labels.csv")



## === cell 2
labels = pd.read_csv(LABELS_CSV)
labels["filename"] = labels["id"] + ".jpg"
classes = sorted(labels["breed"].unique())
num_classes = len(classes)
print("Num classes:", num_classes)



## === cell 3
train_df, val_df = train_test_split(
    labels, test_size=0.2, stratify=labels["breed"], random_state=42
)

TMP_DIR = "/root/tmp_dog_data"
TMP_TRAIN = os.path.join(TMP_DIR, "train")
TMP_VAL = os.path.join(TMP_DIR, "val")
TMP_TEST = os.path.join(TMP_DIR, "test")
for d in [TMP_TRAIN, TMP_VAL, TMP_TEST]:
    os.makedirs(d, exist_ok=True)
    for cls in classes:
        os.makedirs(os.path.join(d, cls), exist_ok=True)

for _, row in train_df.iterrows():
    src = os.path.join(TRAIN_DIR, row["filename"])
    dst = os.path.join(TMP_TRAIN, row["breed"], row["filename"])
    shutil.copyfile(src, dst)

for _, row in val_df.iterrows():
    src = os.path.join(TRAIN_DIR, row["filename"])
    dst = os.path.join(TMP_VAL, row["breed"], row["filename"])
    shutil.copyfile(src, dst)

TEST_SUBDIR = os.path.join(TMP_TEST, "test")
os.makedirs(TEST_SUBDIR, exist_ok=True)
for fname in sorted(os.listdir(TEST_DIR)):
    src_path = os.path.join(TEST_DIR, fname)
    if os.path.isfile(src_path):
        shutil.copyfile(src_path, os.path.join(TEST_SUBDIR, fname))




## === cell 4
def load_images_from_dir(root_dir, img_size=(64, 64)):
    X, y = [], []
    for breed in os.listdir(root_dir):
        breed_dir = os.path.join(root_dir, breed)
        if not os.path.isdir(breed_dir):
            continue
        for fname in os.listdir(breed_dir):
            img_path = os.path.join(breed_dir, fname)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, img_size)
            X.append(img.flatten())
            y.append(breed)
    X = np.array(X, dtype=np.float32) / 255.0
    y = np.array(y)
    return X, y




## === cell 5
print("Loading training data...")
X_train, y_train = load_images_from_dir(TMP_TRAIN)
print("Loading validation data...")
X_val, y_val = load_images_from_dir(TMP_VAL)



## === cell 6
le = LabelEncoder()
le.fit(classes)
y_train_enc = le.transform(y_train)
y_val_enc = le.transform(y_val)

model = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    max_iter=500,
    C=0.5,
    class_weight="balanced",
    n_jobs=-1,
    verbose=0,
)
print("Training Logistic Regression...")
model.fit(X_train, y_train_enc)

val_acc = model.score(X_val, y_val_enc)
val_logloss = log_loss(y_val_enc, model.predict_proba(X_val))
print(f"Validation accuracy: {val_acc:.4f}")
print(f"Validation Log‑Loss: {val_logloss:.5f}")




## === cell 7
def load_test_images(test_dir, img_size=(64, 64)):
    ids = []
    X = []
    for fname in sorted(os.listdir(test_dir)):
        img_path = os.path.join(test_dir, fname)
        if not os.path.isfile(img_path):
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, img_size)
        X.append(img.flatten())
        ids.append(os.path.splitext(fname)[0])
    X = np.array(X, dtype=np.float32) / 255.0
    return ids, X


print("Loading test data...")
test_ids, X_test = load_test_images(TEST_SUBDIR)

print("Predicting probabilities on test set...")
test_proba = model.predict_proba(X_test)  # shape (n_test, n_classes)



## === cell 8
pred_df = pd.DataFrame(test_proba, columns=le.classes_)
pred_df.insert(0, "id", test_ids)



## === cell 9
sample_sub = pd.read_csv(SAMPLE_SUBMIT, nrows=0)  # header only
ordered_cols = ["id"] + [c for c in sample_sub.columns if c != "id"]
submission = pred_df[ordered_cols]

submission_path = "./submission.csv"
submission.to_csv(submission_path, index=False)
print("Submission saved to", submission_path)
