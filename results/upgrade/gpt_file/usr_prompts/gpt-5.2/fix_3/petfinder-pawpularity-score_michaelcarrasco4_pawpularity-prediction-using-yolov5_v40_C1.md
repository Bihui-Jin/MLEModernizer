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

geopandas==0.14.4
imageio==2.37.0
imageio-ffmpeg==0.6.0
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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
tqdm==4.67.1

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
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import math
from tqdm.auto import tqdm
import imageio.v2 as imageio
import torch  # kept (original import), not required after YOLO removal
import matplotlib.patches as patches  # kept (original import), not required after YOLO removal
import os
import matplotlib.image as img
import warnings
import hashlib
import pickle
from concurrent.futures import ProcessPoolExecutor

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import explained_variance_score
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn import svm
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn import datasets

digits = datasets.load_digits()
from sklearn.naive_bayes import BernoulliNB

RANDOM_STATE = 7

DATA_DIR = "../input/petfinder-pawpularity-score"
TRAIN_CSV = f"{DATA_DIR}/train.csv"
TEST_CSV = f"{DATA_DIR}/test.csv"
TRAIN_IMG_DIR = f"{DATA_DIR}/train"
TEST_IMG_DIR = f"{DATA_DIR}/test"

DO_PLOTS = False

CACHE_DIR = "./cache_petfinder"
os.makedirs(CACHE_DIR, exist_ok=True)




## === cell 1
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)

print(train.shape, test.shape)
train.head()




## === cell 2
if DO_PLOTS:
    plt.figure(figsize=(15, 15))
    numeric_corr = train.select_dtypes(include=[np.number]).corr()
    sns.heatmap(numeric_corr, annot=True, fmt=".1g")
    plt.title("Correlation Matrix", fontweight="bold", fontsize=20)
    plt.show()




## === cell 3
if DO_PLOTS:
    train.hist(column="Pawpularity", bins=20)




## === cell 4
if DO_PLOTS:
    train.hist(column="Pawpularity", by="Blur", bins=10)
    plt.xlabel("Blur")
    plt.ylabel("Pawpularity")




## === cell 5
if DO_PLOTS:
    train.hist(column="Pawpularity", by="Near", bins=10)
    plt.xlabel("Near")
    plt.ylabel("Pawpularity")




## === cell 6
if DO_PLOTS:
    train.hist(column="Pawpularity", by="Group", bins=10)
    plt.xlabel("Group")
    plt.ylabel("Pawpularity")




## === cell 7
if DO_PLOTS:
    train.hist(column="Pawpularity", by="Near", bins=10)
    plt.xlabel("Near")
    plt.ylabel("Pawpularity")




## === cell 8
if DO_PLOTS:
    train_image = TRAIN_IMG_DIR
    top_images = train.sort_values(by="Pawpularity", ascending=False)
    top_images = top_images["Id"][:6]

    fig = plt.figure(figsize=(30, 30))
    fig.suptitle("Top 6 Pawpularity Score", fontsize=80)
    for i in range(0, 6):
        image = img.imread(os.path.join(train_image, list(top_images)[i] + ".jpg"))
        _ = fig.add_subplot(2, 3, i + 1)
        plt.imshow(image)
    plt.show()




## === cell 9
if DO_PLOTS:
    train_image = TRAIN_IMG_DIR
    worst_images = train.sort_values(by="Pawpularity", ascending=True)
    worst_images = worst_images["Id"][:6]
    fig = plt.figure(figsize=(30, 30))
    fig.suptitle("Bottom 6 Pawpularity Score", fontsize=80)
    for i in range(0, 6):
        image = img.imread(os.path.join(train_image, list(worst_images)[i] + ".jpg"))
        _ = fig.add_subplot(2, 3, i + 1)
        plt.imshow(image)
    plt.show()




## === cell 10
def get_image_file_path_train(image_id: str) -> str:
    return f"{TRAIN_IMG_DIR}/{image_id}.jpg"


def get_image_file_path_test(image_id: str) -> str:
    return f"{TEST_IMG_DIR}/{image_id}.jpg"


def get_image_info(file_path: str, plot: bool = False):
    """
    Replaces broken YOLO-based feature extraction with image statistics.
    Keeps downstream structure similar: returns dict with 'n_pets','label','pet_ratio'
    plus additional stats to improve model signal.
    """
    image = imageio.imread(file_path)
    if image.ndim == 2:  # grayscale safeguard
        image = np.stack([image, image, image], axis=-1)

    h, w = image.shape[:2]
    img_f = image.astype(np.float32) / 255.0

    mean_rgb = img_f.mean(axis=(0, 1))
    std_rgb = img_f.std(axis=(0, 1))
    mean_all = float(img_f.mean())
    std_all = float(img_f.std())

    gx = np.abs(np.diff(img_f, axis=1)).mean()
    gy = np.abs(np.diff(img_f, axis=0)).mean()
    grad = float((gx + gy) / 2.0)

    info = {
        "n_pets": 1,
        "label": "unknown",
        "pet_ratio": 0.0,
        "img_h": float(h),
        "img_w": float(w),
        "img_aspect": float(w / max(h, 1)),
        "mean_r": float(mean_rgb[0]),
        "mean_g": float(mean_rgb[1]),
        "mean_b": float(mean_rgb[2]),
        "std_r": float(std_rgb[0]),
        "std_g": float(std_rgb[1]),
        "std_b": float(std_rgb[2]),
        "mean_all": mean_all,
        "std_all": std_all,
        "grad": grad,
    }

    if plot:
        plt.figure(figsize=(6, 6))
        plt.title(f"Image preview\n{os.path.basename(file_path)}", fontsize=12)
        plt.imshow(image)
        plt.axis("off")
        plt.show()

    return info


def _cache_key_from_paths(paths, extra_tag="v1"):
    h = hashlib.md5()
    h.update(extra_tag.encode("utf-8"))
    h.update(str(len(paths)).encode("utf-8"))
    for p in paths:
        h.update(p.encode("utf-8"))
    return h.hexdigest()


def extract_image_features_cached(file_paths, cache_path, max_workers=None):
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            feats = pickle.load(f)
        return feats

    if max_workers is None:
        cpu = os.cpu_count() or 2
        max_workers = min(8, max(2, cpu // 2))

    feats = [None] * len(file_paths)
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        feats = list(
            tqdm(
                ex.map(get_image_info, file_paths),
                total=len(file_paths),
                desc="Extracting image features",
            )
        )
    with open(cache_path, "wb") as f:
        pickle.dump(feats, f, protocol=pickle.HIGHEST_PROTOCOL)
    return feats




## === cell 11
train["file_path"] = TRAIN_IMG_DIR + "/" + train["Id"].astype(str) + ".jpg"
test["file_path"] = TEST_IMG_DIR + "/" + test["Id"].astype(str) + ".jpg"

print(train["file_path"].iloc[0])
print(test["file_path"].iloc[0])




## === cell 12
train_paths = train["file_path"].tolist()
train_cache = os.path.join(
    CACHE_DIR, f"train_feats_{_cache_key_from_paths(train_paths)}.pkl"
)
train_image_features = extract_image_features_cached(train_paths, train_cache)

train_img_df = pd.DataFrame(train_image_features)
train = pd.concat([train, train_img_df], axis=1)
train.head()




## === cell 13
if DO_PLOTS:
    for fp in train["file_path"].head(3):
        _ = get_image_info(fp, plot=True)




## === cell 14
train["label"] = train["label"].replace({"dog": 0, "cat": 1, "unknown": 2}).astype(int)

DOG_MEAN = train.loc[train["label"] == 0, "Pawpularity"].mean()
CAT_MEAN = train.loc[train["label"] == 1, "Pawpularity"].mean()
N_MEAN = train.loc[train["label"] == 2, "Pawpularity"].mean()
print("DOG_MEAN:", DOG_MEAN, "CAT_MEAN:", CAT_MEAN, "UNKNOWN_MEAN:", N_MEAN)




## === cell 15
if DO_PLOTS:
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 2])
    langs = ["Dog", "Cat", "Unknown"]
    species = [
        DOG_MEAN if not np.isnan(DOG_MEAN) else N_MEAN,
        CAT_MEAN if not np.isnan(CAT_MEAN) else N_MEAN,
        N_MEAN,
    ]
    ax.bar(langs, species)
    ax.set_ylabel("Mean", fontsize=20)
    ax.set_xlabel("Species", fontsize=20)
    plt.show()




## === cell 16
if DO_PLOTS:
    plt.figure(figsize=(15, 8))
    plt.title("Pawpularity Distribution", size=24)
    train.groupby("label")["Pawpularity"].plot(kind="hist", bins=20, alpha=0.50)
    plt.legend(prop={"size": 20})




## === cell 17
pd.set_option("display.max_columns", None)

drop_cols = ["Id", "file_path"]
target_col = "Pawpularity"

train2 = train.drop(columns=drop_cols)
X = train2.drop(columns=[target_col])
y = train2[target_col].astype(float)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE
)

print(X_train.shape, X_val.shape)




## === cell 18
import matplotlib.patches as mpatches


def ActualvPredictionsGraph(y_test, y_pred, title):
    plt.figure(figsize=(12, 3))
    plt.scatter(range(len(y_test)), y_test, color="blue", s=10)
    plt.scatter(range(len(y_pred)), y_pred, color="red", s=10)
    plt.xlabel("Index ")
    plt.ylabel("Pawpularity ")
    plt.title(title, fontdict={"fontsize": 15})
    plt.legend(
        handles=[
            mpatches.Patch(color="red", label="prediction"),
            mpatches.Patch(color="blue", label="actual"),
        ]
    )
    plt.show()
    return




## === cell 19
results = pd.DataFrame(
    [
        {
            "model": "Decision_tree",
            "best_score": np.nan,
            "best_params": {
                "max_depth": 10,
                "min_samples_leaf": 10,
                "min_samples_split": 5,
            },
        }
    ],
    columns=["model", "best_score", "best_params"],
)
results




## === cell 20
model = tree.DecisionTreeRegressor(
    max_depth=10, min_samples_leaf=10, min_samples_split=5, random_state=RANDOM_STATE
)
model = model.fit(X_train, y_train)
y_pred = model.predict(X_val)

r2 = (r2_score(y_val, y_pred)) * 100
print("R2 score:", r2)
MSE = mean_squared_error(y_val, y_pred)
RMSE = math.sqrt(MSE)
x = RMSE
print("Root Mean Square Error:", RMSE)

if DO_PLOTS:
    ActualvPredictionsGraph(
        y_val.iloc[0:50].values, y_pred[0:50], "Actual vs. Predicted over 50 values"
    )




## === cell 21
if DO_PLOTS:
    fig = plt.figure(figsize=(25, 20))
    _ = tree.plot_tree(model, max_depth=3, filled=True)




## === cell 22
y = np.nan
print("Skipping GaussianNB regression (not applicable to continuous target).")




## === cell 23
z = np.nan
print("Skipping SVR training (not used for submission and may exceed time).")




## === cell 24
if DO_PLOTS:
    names = ["Decision Tree"]
    values = [x]
    plt.title("Root Mean Squared Error")
    plt.ylabel("RMSE Value")
    plt.bar(names, values)




## === cell 25
test_paths = test["file_path"].tolist()
test_cache = os.path.join(
    CACHE_DIR, f"test_feats_{_cache_key_from_paths(test_paths)}.pkl"
)
test_image_features = extract_image_features_cached(test_paths, test_cache)

test_img_df = pd.DataFrame(test_image_features)
test = pd.concat([test, test_img_df], axis=1)

test["label"] = test["label"].replace({"dog": 0, "cat": 1, "unknown": 2}).astype(int)

X_test = test.drop(columns=["Id", "file_path"])
X_test = X_test.reindex(columns=X.columns, fill_value=0)

print(X_test.shape, X.shape)




## === cell 26
final_model = tree.DecisionTreeRegressor(
    max_depth=10, min_samples_leaf=10, min_samples_split=5, random_state=RANDOM_STATE
)
final_model.fit(X, y=train2[target_col].astype(float))

y_new = final_model.predict(X_test)
y_new = np.clip(y_new, 0, 100)




## === cell 27
submission = pd.DataFrame({"Id": test["Id"].values, "Pawpularity": y_new})
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
