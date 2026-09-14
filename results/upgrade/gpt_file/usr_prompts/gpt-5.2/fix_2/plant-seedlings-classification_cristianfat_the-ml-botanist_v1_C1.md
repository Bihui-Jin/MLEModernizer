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
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import seaborn as sns
import os

plt.style.use("seaborn")



## === cell 1
p = "/kaggle/input/plant-seedlings-classification"


def df_of_images(folder_name, path="/kaggle/input/plant-seedlings-classification"):
    itms = []
    folder_path = os.path.join(path, folder_name)
    for cls in sorted(os.listdir(folder_path)):
        cls_path = os.path.join(folder_path, cls)
        if not os.path.isdir(cls_path):
            continue
        for img in sorted(os.listdir(cls_path)):
            img_path = os.path.join(cls_path, img)
            if os.path.isfile(img_path) and img.lower().endswith(
                (".png", ".jpg", ".jpeg")
            ):
                itms.append(
                    {
                        "label": cls.lower()
                        .strip()
                        .replace(" ", "_")
                        .replace("-", "_"),
                        "image_path": img_path,
                    }
                )
    return pd.DataFrame(itms)


train = df_of_images("train")

test_dir = os.path.join(p, "test")
test_files = sorted(
    [
        fn
        for fn in os.listdir(test_dir)
        if os.path.isfile(os.path.join(test_dir, fn))
        and fn.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
)
test = pd.DataFrame({"image_path": [os.path.join(test_dir, fn) for fn in test_files]})

print("train:", train.shape, "test:", test.shape)
print("train labels:", train["label"].nunique())



## === cell 2
train.label.value_counts().plot.bar(rot=0, figsize=(25, 7))



## === cell 3
import tf_keras as keras
from tf_keras.applications.inception_v3 import InceptionV3, preprocess_input
from tf_keras.preprocessing import image
from tf_keras.models import Model



## === cell 4
base_model = InceptionV3(weights="imagenet", include_top=True)
feat_model = Model(
    inputs=base_model.input, outputs=base_model.get_layer("avg_pool").output
)




## === cell 5
def extract_features_keras(image_path, model):
    img = image.load_img(image_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    predictions = model.predict(x, verbose=0)
    return np.squeeze(predictions)




## === cell 6
train["image_features"] = train.image_path.apply(
    lambda x: extract_features_keras(x, feat_model)
)
test["image_features"] = test.image_path.apply(
    lambda x: extract_features_keras(x, feat_model)
)

print("Feature vector length:", len(train["image_features"].iloc[0]))



## === cell 7
from sklearn import metrics
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder



## === cell 8
train_, test_ = train_test_split(
    train, test_size=0.33, random_state=42, stratify=train.label
)

print(
    "train label distribution head:\n",
    (train_.label.value_counts() / len(train_)).head(),
    "\nvalid label distribution head:\n",
    (test_.label.value_counts() / len(test_)).head(),
)



## === cell 9
le = LabelEncoder()
le.fit(train["label"])

X_train = np.vstack(train_["image_features"].values)
y_train = le.transform(train_["label"].values)

X_valid = np.vstack(test_["image_features"].values)
y_valid = le.transform(test_["label"].values)

xgc = xgb.XGBClassifier(
    objective="multi:softmax",
    num_class=train.label.nunique(),
    random_state=42,
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    tree_method="hist",
)
xgc.fit(X_train, y_train)



## === cell 10
results = test_.copy()
valid_pred = xgc.predict(X_valid)
results["y_pred"] = le.inverse_transform(valid_pred)

print(metrics.classification_report(results.label, results.y_pred))



## === cell 11
sns.heatmap(
    metrics.confusion_matrix(results.label, results.y_pred), annot=True, fmt="d"
)



## === cell 12
X_full = np.vstack(train["image_features"].values)
y_full = le.transform(train["label"].values)

xgc_full = xgb.XGBClassifier(
    objective="multi:softmax",
    num_class=train.label.nunique(),
    random_state=42,
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    tree_method="hist",
)
xgc_full.fit(X_full, y_full)

X_test = np.vstack(test["image_features"].values)
test_pred = xgc_full.predict(X_test)
test["species_norm"] = le.inverse_transform(test_pred)

label_map = {
    x.lower().strip().replace(" ", "_").replace("-", "_"): x
    for x in os.listdir(os.path.join(p, "train"))
}

results_sub = pd.DataFrame()
results_sub["file"] = test["image_path"].apply(lambda x: os.path.basename(x))
results_sub["species"] = test["species_norm"].replace(label_map)

sample_path = os.path.join(p, "sample_submission.csv")
sample = pd.read_csv(sample_path)
assert list(sample.columns) == [
    "file",
    "species",
], "Unexpected sample_submission columns"
assert len(results_sub) == len(
    sample
), f"Row mismatch vs sample_submission: {len(results_sub)} vs {len(sample)}"

results_sub.to_csv("submission.csv", index=False)
print(results_sub.head())
print("Wrote submission.csv with shape:", results_sub.shape)
