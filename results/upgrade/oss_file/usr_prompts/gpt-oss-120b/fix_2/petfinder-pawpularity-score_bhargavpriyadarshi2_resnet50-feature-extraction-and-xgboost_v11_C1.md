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
Predict engagement with a pet's profile based on the photograph for that profile.

## Metric
Root mean squared error.

## Submission Format
For each `Id` in the test set, you must predict a probability for the target variable, `Pawpularity`. The file should contain a header and have the following format:

```
Id, Pawpularity
0008dbfb52aa1dc6ee51ee02adf13537, 99.24
0014a7b528f1682f0cf3b73a991c17a0, 61.71
0019c1388dfcd30ac8b112fb4250c251, 6.23
00307b779c82716b240a24f028b0031b, 9.43
00320c6dd5b4223c62a9670110d47911, 70.89
etc.
```

## Dataset
- **train/** - Folder containing training set photos of the form **{id}.jpg**, where **{id}** is a unique Pet Profile ID.
- **train.csv** - Metadata (described below) for each photo in the training set as well as the target, the photo's Pawpularity score. The Id column gives the photo's unique Pet Profile ID corresponding the photo's file name.

The train.csv and test.csv files contain metadata for photos in the training set and test set, respectively. Each pet photo is labeled with the value of 1 (Yes) or 0 (No) for each of the following features:

- **Focus** - Pet stands out against uncluttered background, not too close / far.
- **Eyes** - Both eyes are facing front or near-front, with at least 1 eye / pupil decently clear.
- **Face** - Decently clear face, facing front or near-front.
- **Near** - Single pet taking up significant portion of photo (roughly over 50% of photo width or height).
- **Action** - Pet in the middle of an action (e.g., jumping).
- **Accessory** - Accompanying physical or digital accessory / prop (i.e. toy, digital sticker), excluding collar and leash.
- **Group** - More than 1 pet in the photo.
- **Collage** - Digitally-retouched photo (i.e. with digital photo frame, combination of multiple photos).
- **Human** - Human in the photo.
- **Occlusion** - Specific undesirable objects blocking part of the pet (i.e. human, cage or fence). Note that not all blocking objects are considered occlusion.
- **Info** - Custom-added text or labels (i.e. pet name, description).
- **Blur** - Noticeably out of focus or noisy, especially for the pet's eyes and face. For Blur entries, "Eyes" column is always set to 0.

# 2. Python version

3.10

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (132 lines)
            sample_submission.csv (993 lines)
            sample_submission.csv.zip (22.7 kB)
            test.csv (993 lines)
            test.csv.zip (22.5 kB)
            test.zip (102.2 MB)
            train.csv (8921 lines)
            train.csv.zip (213.0 kB)
            train.zip (926.9 MB)
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
            test/
                a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                ... and 990 other files
                test/
            train/
                e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                ... and 8918 other files
                train/
        input/
            description.md (132 lines)
            sample_submission.csv (993 lines)
            sample_submission.csv.zip (22.7 kB)
            test.csv (993 lines)
            test.csv.zip (22.5 kB)
            test.zip (102.2 MB)
            train.csv (8921 lines)
            train.csv.zip (213.0 kB)
            train.zip (926.9 MB)
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
            test/
                a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                ... and 990 other files
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
            train/
                e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                ... and 8918 other files
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
        working/
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
```

-> data/petfinder-pawpularity-score/sample_submission.csv has 992 rows and 2 columns.
The columns are: Id, Pawpularity

-> data/petfinder-pawpularity-score/test.csv has 992 rows and 13 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur

-> data/petfinder-pawpularity-score/train.csv has 8920 rows and 14 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur, Pawpularity

-> data/sample_submission.csv has 992 rows and 2 columns.
The columns are: Id, Pawpularity

-> data/test.csv has 992 rows and 13 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur

-> data/train.csv has 8920 rows and 14 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur, Pawpularity

-> (stopped after 10 files for performance)

# 5. Target score

18.92990047586468

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, glob
import numpy as np, pandas as pd
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.applications.resnet import ResNet50, preprocess_input
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_img_dir = "../input/petfinder-pawpularity-score/train/"
test_img_dir = "../input/petfinder-pawpularity-score/test/"
train_csv_path = "../input/petfinder-pawpularity-score/train.csv"
test_csv_path = "../input/petfinder-pawpularity-score/test.csv"

train_image_paths = glob.glob(os.path.join(train_img_dir, "*.jpg"))
test_image_paths = glob.glob(os.path.join(test_img_dir, "*.jpg"))




## === cell 2
def load_image(img_path):
    """Load an image, resize to 224×224 and apply ResNet preprocessing."""
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = preprocess_input(img)  # in‑place scaling
    return img




## === cell 3
image_model = ResNet50(include_top=False, weights="imagenet", pooling="avg")




## === cell 4
def generate_features(img_paths):
    """
    Return a dict mapping image Id (filename without extension) to a
    2048‑dim feature vector extracted by ResNet50.
    """
    features = {}
    batch = []
    batch_ids = []
    batch_size = 32

    for p in tqdm(img_paths, desc="Extracting features"):
        img_id = os.path.splitext(os.path.basename(p))[0]
        batch.append(load_image(p))
        batch_ids.append(img_id)

        if len(batch) == batch_size:
            imgs = tf.stack(batch, axis=0)
            feats = image_model(imgs, training=False).numpy()
            for i, f in enumerate(feats):
                features[batch_ids[i]] = f
            batch, batch_ids = [], []

    if batch:
        imgs = tf.stack(batch, axis=0)
        feats = image_model(imgs, training=False).numpy()
        for i, f in enumerate(feats):
            features[batch_ids[i]] = f
    return features




## === cell 5
df_train_meta = pd.read_csv(train_csv_path)
y = df_train_meta["Pawpularity"].values
train_meta_features = df_train_meta.drop(columns=["Id", "Pawpularity"])



## === cell 6
train_img_features = generate_features(train_image_paths)



## === cell 7
train_feat_df = pd.DataFrame.from_dict(train_img_features, orient="index")
train_feat_df.reset_index(inplace=True)
train_feat_df.rename(columns={"index": "Id"}, inplace=True)
df_train = pd.merge(df_train_meta[["Id"]], train_feat_df, on="Id", how="left")
df_train = pd.concat(
    [train_meta_features.reset_index(drop=True), df_train.drop(columns=["Id"])], axis=1
)



## === cell 8
image_feat_cols = list(train_feat_df.columns[1:])  # exclude Id column
pca = PCA(n_components=32, random_state=42)
pca.fit(df_train[image_feat_cols])
train_img_pca = pca.transform(df_train[image_feat_cols])

scaler = StandardScaler()
train_img_scaled = scaler.fit_transform(train_img_pca)

meta_cols = [c for c in df_train.columns if c not in image_feat_cols]
X = np.hstack([df_train[meta_cols].values, train_img_scaled])



## === cell 9
model = XGBRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.085,
    subsample=0.7,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=4,
    objective="reg:squarederror",
)
model.fit(X, y)



## === cell 10
df_test_meta = pd.read_csv(test_csv_path)



## === cell 11
test_img_features = generate_features(test_image_paths)



## === cell 12
test_feat_df = pd.DataFrame.from_dict(test_img_features, orient="index")
test_feat_df.reset_index(inplace=True)
test_feat_df.rename(columns={"index": "Id"}, inplace=True)
df_test = pd.merge(df_test_meta[["Id"]], test_feat_df, on="Id", how="left")
test_img_pca = pca.transform(df_test[image_feat_cols])
test_img_scaled = scaler.transform(test_img_pca)
X_test = np.hstack([df_test[meta_cols].values, test_img_scaled])



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2911878025.py in <cell line: 0>()
      7 test_img_pca = pca.transform(df_test[image_feat_cols])
      8 test_img_scaled = scaler.transform(test_img_pca)
----> 9 X_test = np.hstack([df_test[meta_cols].values, test_img_scaled])
     10 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6247         if nmissing:
   6248             if nmissing == len(indexer):
-> 6249                 raise KeyError(f"None of [{key}] are in the [{axis_name}]")
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())

KeyError: "None of [Index(['Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory', 'Group',\n       'Collage', 'Human', 'Occlusion', 'Info', 'Blur'],\n      dtype='object')] are in the [columns]"

## === cell 13
y_pred = model.predict(X_test)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3275972757.py in <cell line: 0>()
      1 # Predict on test data
----> 2 y_pred = model.predict(X_test)
      3 

NameError: name 'X_test' is not defined

## === cell 14
submission = pd.DataFrame({"Id": df_test_meta["Id"], "Pawpularity": y_pred})
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv, shape:", submission.shape)

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1781459235.py in <cell line: 0>()
      1 # Prepare submission file
----> 2 submission = pd.DataFrame({"Id": df_test_meta["Id"], "Pawpularity": y_pred})
      3 submission.to_csv("submission.csv", index=False)
      4 print("Submission saved to submission.csv, shape:", submission.shape)

NameError: name 'y_pred' is not defined
