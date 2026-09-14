# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Predict the individual whale species in images.

## Metric
Mean Average Precision @ 5 (MAP@5).

## Submission Format
For each `Image` in the test set, you may predict up to 5 labels for the whale `Id`. Whales that are not predicted to be one of the labels in the training data should be labeled as `new_whale`. The file should contain a header and have the following format:

```
Image,Id
00029b3a.jpg,new_whale w_1287fbc w_98baff9 w_7554f44 w_1eafe46
0003c693.jpg,new_whale w_1287fbc w_98baff9 w_7554f44 w_1eafe46
...
```

## Dataset
This training data contains thousands of images of humpback whale flukes. Individual whales have been identified by researchers and given an `Id`. The challenge is to predict the whale `Id` of images in the test set. What makes this such a challenge is that there are only a few examples for each of 3,000+ whale Ids.

- **train.zip** - a folder containing the training images
- **train.csv** - maps the training `Image` to the appropriate whale `Id`. Whales that are not predicted to have a label identified in the training data should be labeled as `new_whale`.
- **test.zip** - a folder containing the test images to predict the whale `Id`
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.6

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (51 lines)
            sample_submission.csv (2611 lines)
            sample_submission.csv.zip (15.5 kB)
            test.zip (72.1 MB)
            train.csv (7241 lines)
            train.csv.zip (68.3 kB)
            train.zip (200.4 MB)
            test/
                11482c0d.jpg (27.8 kB)
                68edd063.jpg (11.5 kB)
                ... and 2608 other files
                test/
            train/
                6fc790ac.jpg (17.1 kB)
                f4bb7bb0.jpg (11.6 kB)
                ... and 7238 other files
                train/
            whale-categorization-playground/
                description.md (51 lines)
                sample_submission.csv (2611 lines)
                ... and 5 other files
                test/
                    11482c0d.jpg (27.8 kB)
                    68edd063.jpg (11.5 kB)
                    ... and 2608 other files
                    test/
                train/
                    6fc790ac.jpg (17.1 kB)
                    f4bb7bb0.jpg (11.6 kB)
                    ... and 7238 other files
                    train/
                whale-categorization-playground/
        input/
            description.md (51 lines)
            sample_submission.csv (2611 lines)
            sample_submission.csv.zip (15.5 kB)
            test.zip (72.1 MB)
            train.csv (7241 lines)
            train.csv.zip (68.3 kB)
            train.zip (200.4 MB)
            test/
                11482c0d.jpg (27.8 kB)
                68edd063.jpg (11.5 kB)
                ... and 2608 other files
                test/
                    11482c0d.jpg (27.8 kB)
                    68edd063.jpg (11.5 kB)
                    ... and 2608 other files
                    test/
            train/
                6fc790ac.jpg (17.1 kB)
                f4bb7bb0.jpg (11.6 kB)
                ... and 7238 other files
                train/
                    6fc790ac.jpg (17.1 kB)
                    f4bb7bb0.jpg (11.6 kB)
                    ... and 7238 other files
                    train/
            whale-categorization-playground/
                description.md (51 lines)
                sample_submission.csv (2611 lines)
                ... and 5 other files
                test/
                    11482c0d.jpg (27.8 kB)
                    68edd063.jpg (11.5 kB)
                    ... and 2608 other files
                    test/
                train/
                    6fc790ac.jpg (17.1 kB)
                    f4bb7bb0.jpg (11.6 kB)
                    ... and 7238 other files
                    train/
                whale-categorization-playground/
        working/
            whale-categorization-playground/
                description.md (51 lines)
                sample_submission.csv (2611 lines)
                ... and 5 other files
                test/
                    11482c0d.jpg (27.8 kB)
                    68edd063.jpg (11.5 kB)
                    ... and 2608 other files
                    test/
                train/
                    6fc790ac.jpg (17.1 kB)
                    f4bb7bb0.jpg (11.6 kB)
                    ... and 7238 other files
                    train/
                whale-categorization-playground/
```

-> data/sample_submission.csv has 2610 rows and 2 columns.
The columns are: Image, Id

-> data/train.csv has 7240 rows and 2 columns.
The columns are: Image, Id

-> data/whale-categorization-playground/sample_submission.csv has 2610 rows and 2 columns.
The columns are: Image, Id

-> data/whale-categorization-playground/train.csv has 7240 rows and 2 columns.
The columns are: Image, Id

-> input/sample_submission.csv has 2610 rows and 2 columns.
The columns are: Image, Id

-> input/train.csv has 7240 rows and 2 columns.
The columns are: Image, Id

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
from os.path import split
from subprocess import check_output
import os
from concurrent.futures import ProcessPoolExecutor

NUM_WORKERS = min(8, os.cpu_count() or 1)

print(check_output(["ls", "../input"]).decode("utf8"))



## === cell 1
train_images = sorted(glob("../input/train/*jpg"))
test_images = sorted(glob("../input/test/*jpg"))
df = pd.read_csv("../input/train.csv")

df["Image"] = df["Image"].map(lambda x: "../input/train/" + x)
ImageToLabelDict = dict(zip(df["Image"], df["Id"]))



## === cell 2
SIZE = 64


def ImportImage(filename):
    img = Image.open(filename).convert("L").resize((SIZE, SIZE))
    return np.array(img, dtype=np.uint8)  # keep as uint8 to reduce memory/copy overhead


n_train = len(train_images)
train_img = np.empty((n_train, SIZE, SIZE), dtype=np.uint8)

with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
    for idx, arr in enumerate(executor.map(ImportImage, train_images, chunksize=500)):
        train_img[idx] = arr

x = train_img
print(f"{x.shape[0]} training images")

print("Nbr of samples/class\tNbr of classes")
for idx, val in df["Id"].value_counts().value_counts().sort_index().items():
    print(f"{idx}\t\t\t{val}")




## === cell 3
class LabelOneHotEncoder:
    def __init__(self):
        self.le = LabelEncoder()

    def fit_transform(self, x):
        return self.le.fit_transform(x)

    def transform(self, x):
        return self.le.transform(x)

    def inverse_transform(self, x):
        return self.le.inverse_transform(x)


y = list(map(ImageToLabelDict.get, train_images))
lohe = LabelOneHotEncoder()
y_int = lohe.fit_transform(y)



## === cell 4
x_flat = x.reshape(len(x), -1).astype(np.float32)
scaler = StandardScaler(copy=False)  # avoid an extra copy
x_scaled = scaler.fit_transform(x_flat)



## === cell 5
num_classes = len(np.unique(y_int))
print("Number of classes:", num_classes)

model = LogisticRegression(
    max_iter=300,
    tol=1e-3,
    solver="sag",  # faster dense solver with same multinomial objective
    multi_class="multinomial",
    random_state=42,
    n_jobs=NUM_WORKERS,
    verbose=0,
)

model.fit(x_scaled, y_int)
print("Training completed.")



## === cell 6
warnings.filterwarnings("ignore", category=DeprecationWarning)

n_test = len(test_images)
test_img = np.empty((n_test, SIZE, SIZE), dtype=np.uint8)

with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
    for idx, arr in enumerate(executor.map(ImportImage, test_images, chunksize=500)):
        test_img[idx] = arr

test_flat = test_img.reshape(len(test_img), -1).astype(np.float32)
test_scaled = scaler.transform(test_flat)

probs_batch = model.predict_proba(test_scaled)  # shape (n_test, n_classes)

partition_idx = np.argpartition(probs_batch, -5, axis=1)[:, -5:]
top5_idx_batch = np.take_along_axis(
    partition_idx,
    np.argsort(-probs_batch[np.arange(n_test)[:, None], partition_idx], axis=1),
    axis=1,
)

top5_labels_batch = lohe.inverse_transform(top5_idx_batch)  # shape (n_test, 5)

with open("sample_submission.csv", "w") as f:
    f.write("Image,Id\n")
    for img_path, top_labels in zip(test_images, top5_labels_batch):
        top5_str = " ".join(top_labels)
        image_name = split(img_path)[-1]
        f.write(f"{image_name},{top5_str}\n")
