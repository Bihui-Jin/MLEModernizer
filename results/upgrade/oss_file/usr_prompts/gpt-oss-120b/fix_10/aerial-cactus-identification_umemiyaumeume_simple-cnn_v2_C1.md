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
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.7

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tf_keras==2.18.0
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 3 other files
        input/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 3 other files
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 3 other files
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
Here is some information about the columns:
has_cactus (float64) has 1 unique values: [0.5]
id (object) has 2000 unique values. Some example values: ['09034a34de0e2015a8a28dfe18f423f6.jpg', '44b974fd955c4cd306c7dbae154646b9.jpg', '37cd593f40ba13982e7587a613a9a789.jpg', 'c892c865a5706d3072ba44beafbf2d1c.jpg']

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
Here is some information about the columns:
has_cactus (int64) has 2 unique values: [1, 0]
id (object) has 2000 unique values. Some example values: ['2de8f189f1dce439766637e75df0ee27.jpg', 'f94b18b4c6850fc44b54f5fbe3c5b0c6.jpg', 'fb0624c0ebc01643a8f4318537099078.jpg', '00b4dfbb267109b5f0d0dde365fa6161.jpg']

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
Here is some information about the columns:
has_cactus (float64) has 1 unique values: [0.5]
id (object) has 2000 unique values. Some example values: ['09034a34de0e2015a8a28dfe18f423f6.jpg', '44b974fd955c4cd306c7dbae154646b9.jpg', '37cd593f40ba13982e7587a613a9a789.jpg', 'c892c865a5706d3072ba44beafbf2d1c.jpg']

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
Here is some information about the columns:
has_cactus (int64) has 2 unique values: [1, 0]
id (object) has 2000 unique values. Some example values: ['2de8f189f1dce439766637e75df0ee27.jpg', 'f94b18b4c6850fc44b54f5fbe3c5b0c6.jpg', 'fb0624c0ebc01643a8f4318537099078.jpg', '00b4dfbb267109b5f0d0dde365fa6161.jpg']

-> working/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
Here is some information about the columns:
has_cactus (float64) has 1 unique values: [0.5]
id (object) has 2000 unique values. Some example values: ['09034a34de0e2015a8a28dfe18f423f6.jpg', '44b974fd955c4cd306c7dbae154646b9.jpg', '37cd593f40ba13982e7587a613a9a789.jpg', 'c892c865a5706d3072ba44beafbf2d1c.jpg']

-> working/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
Here is some information about the columns:
has_cactus (int64) has 2 unique values: [1, 0]
id (object) has 2000 unique values. Some example values: ['2de8f189f1dce439766637e75df0ee27.jpg', 'f94b18b4c6850fc44b54f5fbe3c5b0c6.jpg', 'fb0624c0ebc01643a8f4318537099078.jpg', '00b4dfbb267109b5f0d0dde365fa6161.jpg']

# 5. Target score

0.9982

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, sys, numpy as np, pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from PIL import Image

POSSIBLE_ROOTS = [
    "/kaggle/input/aerial-cactus-identification",
    "/kaggle/data/aerial-cactus-identification",  # added
    "../input/aerial-cactus-identification",
    "./aerial-cactus-identification",
    "./working/aerial-cactus-identification",
    "./data/aerial-cactus-identification",
    "./input/aerial-cactus-identification",
    "/kaggle/working/aerial-cactus-identification",
]


def find_data_root():
    """Return the first directory that contains train.csv and a train folder."""
    for root in POSSIBLE_ROOTS:
        root_path = os.path.abspath(root)
        train_csv = os.path.join(root_path, "train.csv")
        train_dir = os.path.join(root_path, "train")
        if (
            os.path.isdir(root_path)
            and os.path.isfile(train_csv)
            and os.path.isdir(train_dir)
        ):
            return root_path

    start_dir = os.path.abspath(".")
    for dirpath, dirnames, filenames in os.walk(start_dir):
        if "train.csv" in filenames and "train" in dirnames:
            return os.path.abspath(dirpath)

    raise FileNotFoundError(
        "Could not locate the aerial-cactus-identification data folder."
    )


DATA_ROOT = find_data_root()
print("Using data root:", DATA_ROOT)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2212332804.py in <cell line: 0>()
     43 
     44 
---> 45 DATA_ROOT = find_data_root()
     46 print("Using data root:", DATA_ROOT)
     47 

/tmp/ipykernel_11/2212332804.py in find_data_root()
     38             return os.path.abspath(dirpath)
     39 
---> 40     raise FileNotFoundError(
     41         "Could not locate the aerial-cactus-identification data folder."
     42     )

FileNotFoundError: Could not locate the aerial-cactus-identification data folder.

## === cell 1
def load_image(path):
    """Load an image as a float32 numpy array scaled to [0, 1]."""
    img = Image.open(path).convert("RGB")  # ensure 3 channels
    return np.asarray(img, dtype="float32") / 255.0




## === cell 2
class DataHandler:
    def __init__(self, csv_name):
        self.dataset_path = DATA_ROOT
        csv_path = os.path.join(self.dataset_path, csv_name)
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(
                f"Could not locate {csv_name} in {self.dataset_path}"
            )
        self.train_data = pd.read_csv(csv_path)

    def _load_images(self, folder):
        """Load all *.jpg images from a folder, returning array (N,H,W,C) and list of filenames."""
        folder_path = os.path.join(self.dataset_path, folder)
        img_names = [f for f in os.listdir(folder_path) if f.lower().endswith(".jpg")]
        datas = []
        for img_name in tqdm(img_names, desc=f"Loading {folder}"):
            img_path = os.path.join(folder_path, img_name)
            datas.append(load_image(img_path))
        return np.asarray(datas, dtype="float32"), img_names

    def get_data_label(self, folname="train", val_rate=0.2):
        """
        Returns (train_imgs, train_labels, val_imgs, val_labels) for training,
        or test_imgs for the test folder (self.test_names is set).
        """
        if folname == "train":
            folder_path = os.path.join(self.dataset_path, "train")
            datas = []
            labels = []
            for _, row in tqdm(
                self.train_data.iterrows(),
                total=len(self.train_data),
                desc="Preparing train",
            ):
                img_name = row["id"]
                img_path = os.path.join(folder_path, img_name)
                datas.append(load_image(img_path))
                labels.append(int(row["has_cactus"]))
            datas = np.asarray(datas, dtype="float32")
            labels = np.asarray(labels, dtype="int")
            x_tr, x_val, y_tr, y_val = train_test_split(
                datas, labels, test_size=val_rate, random_state=42, stratify=labels
            )
            return x_tr, y_tr, x_val, y_val
        else:  # test
            test_imgs, test_names = self._load_images(folname)
            self.test_names = test_names
            return test_imgs

    def get_shape(self, folname="train"):
        """Return the shape of a single image (H, W, C)."""
        folder_path = os.path.join(self.dataset_path, folname)
        sample_files = [
            f for f in os.listdir(folder_path) if f.lower().endswith(".jpg")
        ]
        if not sample_files:
            raise FileNotFoundError(f"No jpg files found in {folder_path}")
        sample_path = os.path.join(folder_path, sample_files[0])
        img = load_image(sample_path)
        return img.shape




## === cell 3
handler = DataHandler("train.csv")
img_shape = handler.get_shape("train")
print("Image shape:", img_shape)

test_imgs = handler.get_data_label("test")
test_names = handler.test_names
print("Test set size:", test_imgs.shape[0], "names collected:", len(test_names))

x_train, y_train, x_val, y_val = handler.get_data_label("train")
print("Training set:", x_train.shape, y_train.shape)
print("Validation set:", x_val.shape, y_val.shape)

x_train_flat = x_train.reshape((x_train.shape[0], -1))
x_val_flat = x_val.reshape((x_val.shape[0], -1))
test_flat = test_imgs.reshape((test_imgs.shape[0], -1))

gbc = GradientBoostingClassifier(
    n_estimators=800,
    learning_rate=0.05,
    max_depth=5,
    random_state=42,
)
gbc.fit(x_train_flat, y_train)

val_pred = gbc.predict_proba(x_val_flat)[:, 1]
val_auc = roc_auc_score(y_val, val_pred)
print(f"Validation AUC: {val_auc:.5f}")

test_pred = gbc.predict_proba(test_flat)[:, 1]
print("Predictions shape:", test_pred.shape)

submission_df = pd.DataFrame({"id": test_names, "has_cactus": test_pred})
submission_df = submission_df[["id", "has_cactus"]]
print(submission_df.head(10))

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/672682360.py in <cell line: 0>()
----> 1 handler = DataHandler("train.csv")
      2 img_shape = handler.get_shape("train")
      3 print("Image shape:", img_shape)
      4 
      5 # Load test images

/tmp/ipykernel_11/2065491051.py in __init__(self, csv_name)
      1 class DataHandler:
      2     def __init__(self, csv_name):
----> 3         self.dataset_path = DATA_ROOT
      4         csv_path = os.path.join(self.dataset_path, csv_name)
      5         if not os.path.isfile(csv_path):

NameError: name 'DATA_ROOT' is not defined
