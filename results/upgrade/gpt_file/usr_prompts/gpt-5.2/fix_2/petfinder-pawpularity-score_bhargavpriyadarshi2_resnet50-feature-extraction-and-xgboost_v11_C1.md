# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras

from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)



## === cell 1
BASE_PATH = "../input/petfinder-pawpularity-score"
TRAIN_CSV_PATH = f"{BASE_PATH}/train.csv"
TEST_CSV_PATH = f"{BASE_PATH}/test.csv"
TRAIN_IMG_GLOB = f"{BASE_PATH}/train/*.jpg"
TEST_IMG_GLOB = f"{BASE_PATH}/test/*.jpg"

train_image = glob.glob(TRAIN_IMG_GLOB)
test_image = glob.glob(TEST_IMG_GLOB)

assert len(train_image) > 0, f"No train images found at {TRAIN_IMG_GLOB}"
assert len(test_image) > 0, f"No test images found at {TEST_IMG_GLOB}"




## === cell 2
def load_file(img_path):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.keras.layers.Resizing(224, 224)(img)
    img = tf.keras.applications.resnet.preprocess_input(img)
    return img, img_path




## === cell 3
image_model = tf.keras.applications.ResNet50(include_top=True, weights="imagenet")
new_input = image_model.input
layer = image_model.layers[-2].output  # penultimate layer (2048-d features)
image_features_extract_model = tf.keras.Model(new_input, layer)

_ = image_features_extract_model(tf.zeros((1, 224, 224, 3), dtype=tf.float32))




## === cell 4
def generate_features(img_name_vector):
    encode_paths = sorted(set(img_name_vector))
    image_dataset = tf.data.Dataset.from_tensor_slices(encode_paths)
    image_dataset = image_dataset.map(
        load_file, num_parallel_calls=tf.data.AUTOTUNE
    ).batch(8)

    extracted_features = {}
    for img, path in tqdm(image_dataset, total=int(np.ceil(len(encode_paths) / 8))):
        batch_features = image_features_extract_model(img, training=False)
        for p, bf in zip(path, batch_features):
            img_id = os.path.basename(p.numpy().decode("utf-8")).split(".")[0]
            extracted_features[img_id] = bf.numpy().astype(np.float32)
    return extracted_features




## === cell 5
temp_train = pd.read_csv(TRAIN_CSV_PATH)
temp_test = pd.read_csv(TEST_CSV_PATH)

y = temp_train["Pawpularity"].values.astype(np.float32)

meta_cols = [
    "Subject Focus",
    "Eyes",
    "Face",
    "Near",
    "Action",
    "Accessory",
    "Group",
    "Collage",
    "Human",
    "Occlusion",
    "Info",
    "Blur",
]

missing_train = [
    c for c in meta_cols + ["Id", "Pawpularity"] if c not in temp_train.columns
]
missing_test = [c for c in meta_cols + ["Id"] if c not in temp_test.columns]
assert not missing_train, f"Missing columns in train.csv: {missing_train}"
assert not missing_test, f"Missing columns in test.csv: {missing_test}"



## === cell 6

extracted_features_train = generate_features(train_image)
extracted_features_test = generate_features(test_image)

df_train_img = pd.DataFrame.from_dict(extracted_features_train, orient="index")
df_train_img.index.name = "Id"
df_train_img = df_train_img.reset_index()

df_test_img = pd.DataFrame.from_dict(extracted_features_test, orient="index")
df_test_img.index.name = "Id"
df_test_img = df_test_img.reset_index()

df_train_merged = pd.merge(
    temp_train[["Id"] + meta_cols + ["Pawpularity"]], df_train_img, on="Id", how="left"
)
df_test_merged = pd.merge(
    temp_test[["Id"] + meta_cols], df_test_img, on="Id", how="left"
)

img_feat_cols = [
    c
    for c in df_train_merged.columns
    if c not in (["Id"] + meta_cols + ["Pawpularity"])
]
df_train_merged[img_feat_cols] = df_train_merged[img_feat_cols].fillna(0.0)
df_test_merged[img_feat_cols] = df_test_merged[img_feat_cols].fillna(0.0)

df_normal_feature = df_train_merged[meta_cols].astype(np.float32)
df_extracted_feature = df_train_merged[img_feat_cols].astype(np.float32)

df_test_normal_feature = df_test_merged[meta_cols].astype(np.float32)
df_test_extracted_feature = df_test_merged[img_feat_cols].astype(np.float32)



## === cell 7
pca = PCA(n_components=32, random_state=SEED)
pca.fit(df_extracted_feature)

df_new_train = pca.transform(df_extracted_feature)
df_new_test = pca.transform(df_test_extracted_feature)



## === cell 8
scaler = StandardScaler()
df_new_train = scaler.fit_transform(df_new_train)
df_new_test = scaler.transform(df_new_test)

df_extracted_feature = pd.DataFrame(df_new_train)
df_test_extracted_feature = pd.DataFrame(df_new_test)



## === cell 9
df_pca_train = pd.concat(
    [
        df_normal_feature.reset_index(drop=True),
        df_extracted_feature.reset_index(drop=True),
    ],
    axis=1,
)
df_pca_test = pd.concat(
    [
        df_test_normal_feature.reset_index(drop=True),
        df_test_extracted_feature.reset_index(drop=True),
    ],
    axis=1,
)

X = df_pca_train.values.astype(np.float32)



## === cell 10
model = XGBRegressor(
    n_estimators=50,
    max_depth=5,
    eta=0.085,
    subsample=0.7,
    colsample_bytree=0.8,
    random_state=SEED,
    n_jobs=4,
)

model.fit(X, y)



## === cell 11
y_pred = model.predict(df_pca_test.values.astype(np.float32))

y_pred = np.clip(y_pred, 1.0, 100.0)

submission = pd.DataFrame(
    {"Id": temp_test["Id"].values, "Pawpularity": y_pred.astype(np.float32)}
)
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
