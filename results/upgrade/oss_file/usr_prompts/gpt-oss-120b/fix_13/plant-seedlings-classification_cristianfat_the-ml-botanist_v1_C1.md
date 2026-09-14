# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Classify plant seedlings into their respective species.

## Metric
Micro-averaged F1-score.

## Submission Format
For each `file` in the test set, you must predict a probability for the `species` variable. The file should contain a header and have the following format:

```
file,species
0021e90e4.png,Maize
003d61042.png,Sugar beet
007b3da8b.png,Common wheat
etc.
```

## Dataset
The list of species is as follows:

```
Black-grass
Charlock
Cleavers
Common Chickweed
Common wheat
Fat Hen
Loose Silky-bent
Maize
Scentless Mayweed
Shepherds Purse
Small-flowered Cranesbill
Sugar beet
```

- **train.csv** - the training set, with plant species organized by folder
- **test.csv** - the test set, you need to predict the species of each image
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
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
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tf_keras==2.18.0
xgboost==2.0.3

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (84 lines)
            sample_submission.csv (667 lines)
            sample_submission.csv.zip (4.6 kB)
            test.zip (259.0 MB)
            train.zip (1.5 GB)
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
            test/
                5db43df54.png (177.5 kB)
                09d34fe5b.png (156.0 kB)
                ... and 664 other files
                test/
            train/
                Black-grass/
                    2ed589264.png (44.6 kB)
                    840a7ed59.png (708.1 kB)
                    ... and 219 other files
                Charlock/
                    ee4a02bf9.png (229.3 kB)
                    e795c53c9.png (354.4 kB)
                    ... and 322 other files
                ... and 11 other folders
        input/
            description.md (84 lines)
            sample_submission.csv (667 lines)
            sample_submission.csv.zip (4.6 kB)
            test.zip (259.0 MB)
            train.zip (1.5 GB)
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
            test/
                5db43df54.png (177.5 kB)
                09d34fe5b.png (156.0 kB)
                ... and 664 other files
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
            train/
                Black-grass/
                    2ed589264.png (44.6 kB)
                    840a7ed59.png (708.1 kB)
                    ... and 219 other files
                Charlock/
                    ee4a02bf9.png (229.3 kB)
                    e795c53c9.png (354.4 kB)
                    ... and 322 other files
                ... and 11 other folders
        working/
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
```

-> data/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> data/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> input/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> input/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> working/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import xgboost as xgb
from concurrent.futures import (
    ProcessPoolExecutor,
)  # use processes for true CPU parallelism
from PIL import Image

np.random.seed(42)




## === cell 1
p = "/kaggle/input/plant-seedlings-classification"


def df_of_images(folder_name, path=p):
    items = []
    for label in sorted(os.listdir(os.path.join(path, folder_name))):
        label_dir = os.path.join(path, folder_name, label)
        if not os.path.isdir(label_dir):
            continue
        for img_name in sorted(os.listdir(label_dir)):
            img_path = os.path.join(label_dir, img_name)
            if not os.path.isfile(img_path):
                continue
            items.append(
                {
                    "label": label.lower().strip().replace(" ", "_").replace("-", "_"),
                    "image_path": img_path,
                }
            )
    return pd.DataFrame(items)


train = df_of_images("train")

test_images = [
    os.path.join(p, "test", f)
    for f in sorted(os.listdir(os.path.join(p, "test")))
    if os.path.isfile(os.path.join(p, "test", f))
]
test = pd.DataFrame({"image_path": test_images})




## === cell 2
IMG_SIZE = 128  # reduced size for speed while keeping the same preprocessing idea


def extract_features(image_path):
    """
    Load an image, resize to IMG_SIZE×IMG_SIZE, scale to [0,1],
    and flatten to a 1‑D float32 vector.
    """
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.ravel()




## === cell 3
def parallel_features(paths):
    """
    Extract features for a list of image paths using a process pool.
    Returns a pre‑allocated 2‑D NumPy array (n_samples × n_features).
    """
    n = len(paths)
    feat_dim = IMG_SIZE * IMG_SIZE * 3
    features = np.empty((n, feat_dim), dtype=np.float32)

    max_workers = os.cpu_count() or 1
    chunk_size = max(1, n // (max_workers * 4))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for idx, feat in enumerate(
            executor.map(extract_features, paths, chunksize=chunk_size)
        ):
            features[idx] = feat
    return features


train_features = parallel_features(train["image_path"].tolist())
test_features = parallel_features(test["image_path"].tolist())




## === cell 4
le = LabelEncoder()
train["label_enc"] = le.fit_transform(train["label"])




## === cell 5
train_split, val_split = train_test_split(
    train,
    test_size=0.33,
    random_state=42,
    stratify=train["label_enc"],
)




## === cell 6
X_train = train_features[train_split.index.values]
y_train = train_split["label_enc"].values

xgc = xgb.XGBClassifier(
    objective="multi:softmax",
    num_class=train["label_enc"].nunique(),
    eval_metric="mlogloss",
    use_label_encoder=False,
    n_estimators=400,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    tree_method="hist",
    n_jobs=-1,
)
xgc.fit(X_train, y_train)




## === cell 7
X_val = train_features[val_split.index.values]
y_val = val_split["label_enc"].values
val_pred = xgc.predict(X_val)
print("Validation F1 (micro):", metrics.f1_score(y_val, val_pred, average="micro"))




## === cell 8
test_pred_enc = xgc.predict(test_features)
test["species"] = le.inverse_transform(test_pred_enc)

submission = pd.DataFrame(
    {
        "file": test["image_path"].apply(lambda x: os.path.basename(x)),
        "species": test["species"],
    }
)

submission.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")
