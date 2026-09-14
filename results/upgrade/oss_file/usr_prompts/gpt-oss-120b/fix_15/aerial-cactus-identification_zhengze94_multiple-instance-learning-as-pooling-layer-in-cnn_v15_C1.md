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
pillow==11.3.0
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 5. Target score

0.8015

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.50272) has done: 'The fix loads and preprocesses all images consistently, replaces the failing TensorFlow model with a simple scikit‑learn logistic regression (which avoids the protobuf issue), correctly splits data, evaluates ROC‑AUC, and writes a proper `sample_submission.csv` containing the required `id,has_cactus` columns.'
- What this solution (achieved 0.50377) has done: 'I replace the sharpening‑based features with the original raw pixel values, add standard‑scaler normalization, and give the logistic regression a balanced class weight to improve discrimination. These minimal adjustments keep the overall pipeline (logistic regression on flattened images) unchanged while addressing scaling and class imbalance, which should raise the ROC‑AUC toward the target.'
- What this solution (achieved 0.50497) has done: 'I add the missing use of the sharpening feature by computing sharpened versions of both train and test images, flattening them, and concatenating them with the raw pixel features before scaling and fitting the logistic regression. This small feature‑engineering tweak keeps the same logistic‑regression pipeline while providing richer information, which should raise the ROC‑AUC toward the target.'
- What this solution (achieved 0.50292) has done: 'I fixed the missing `glob` import, corrected the fallback directory logic, and added a safety check so the image‑loading step always succeeds. With these fixes the whole pipeline runs, computes the validation ROC‑AUC, and writes a proper `sample_submission.csv` containing the required `id,has_cactus` columns.'
- What this solution (achieved 0.50517) has done: 'I keep the overall pipeline (image loading, sharpening, splitting, scaling, and logistic‑regression prediction) but replace the huge raw‑pixel feature vector with a compact set of channel‑statistics‑only features (mean and standard‑deviation for each colour channel of the original and sharpened images).  This reduces noise, lets the linear model focus on the colour information that actually distinguishes cactus images, and typically moves the ROC‑AUC much closer to the target without altering the core logic.'
- What this solution (achieved 0.50035) has done: 'The fix replaces the OpenCV Laplacian computation (which caused an unsupported‑format error) with a SciPy‑based Laplacian that works on float images. This resolves the runtime exceptions, restores the feature‑generation pipeline, and enables the later cells (scaling, training, validation, and submission) to execute correctly, producing a valid `sample_submission.csv`.'
- What this solution (achieved 0.50819) has done: 'I add raw‑pixel features to the existing channel‑statistics feature set and slightly increase the regularisation strength (C) of the Logistic Regression model. These extra features give the linear model more expressive power while keeping the overall pipeline (feature extraction → scaling → LogisticRegression) unchanged, which should raise the validation ROC‑AUC toward the target score.'
- What this solution (achieved 0.49411) has done: 'I replace the logistic‑regression pipeline with a lightweight convolutional neural network, because the current ≈0.51 ROC‑AUC is far below the target 0.8015 and the relative gap exceeds 30 %. Keeping the image‑loading steps unchanged, I split the raw 32×32 RGB images into train/validation sets, train a small CNN (three Conv2D layers with pooling, global average pooling and a final dense sigmoid), compute the validation AUC, and use the trained model to predict the test set. The final cells write a proper `sample_submission.csv` with the required columns, preserving the original submission format.'
- What this solution (achieved 0.49202) has done: 'I set the protobuf implementation environment variable before importing TensorFlow to fix the import error, and then improve the model training by adding an AUC metric, using it for early stopping, training for more epochs, and applying class weighting to address label imbalance. These changes keep the overall CNN pipeline intact while fixing the runtime crash and nudging the validation ROC‑AUC toward the target.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import glob
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.utils import class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

np.random.seed(42)




## === cell 1
def load_imgs(path):
    """
    Load all JPEG images from a directory, resize to 32x32, convert to RGB,
    and scale pixel values to [0,1]. Returns a dict {filename: image_array}.
    """
    imgs = {}
    for f in [os.path.basename(p) for p in glob.glob(os.path.join(path, "*.jpg"))]:
        fp = os.path.join(path, f)
        img = cv2.imread(fp, cv2.IMREAD_COLOR)
        if img is None:
            continue  # skip unreadable files
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (32, 32))
        imgs[f] = img.astype(np.float32) / 255.0
    return imgs


train_dir = "../input/aerial-cactus-identification/train"
if not os.path.isdir(train_dir):
    train_dir = "../input/train"
test_dir = "../input/aerial-cactus-identification/test"
if not os.path.isdir(test_dir):
    test_dir = "../input/test"

img_train = load_imgs(train_dir)
img_test = load_imgs(test_dir)

assert len(img_train) > 0, "No training images were loaded."
assert len(img_test) > 0, "No test images were loaded."




## === cell 2
train_csv = pd.read_csv("../input/aerial-cactus-identification/train.csv")
if train_csv.empty:
    train_csv = pd.read_csv("../input/train.csv")

X_train = []
Y_train = []

for _, row in train_csv.iterrows():
    fname = row["id"]
    if fname in img_train:
        X_train.append(img_train[fname])
        Y_train.append(int(row["has_cactus"]))

X_train = np.array(X_train)  # (n_samples, 32, 32, 3)
Y_train = np.array(Y_train)

test_filenames = sorted(img_test.keys())
X_test = np.array([img_test[f] for f in test_filenames])

print("Training data shape:", X_train.shape, "=>", Y_train.shape)
print("Test data shape:", X_test.shape)




## === cell 3
plt.rcParams["axes.grid"] = False




## === cell 4
fig, axes = plt.subplots(1, 5, figsize=(15, 4))
for i, ax in enumerate(axes):
    idx = i * 200
    if idx < len(X_train):
        ax.imshow(X_train[idx])
        ax.set_title("Has cactus: " + str(Y_train[idx]))
plt.show()




## === cell 5
def img_sharpen(img):
    blurred_f = cv2.GaussianBlur(img, (5, 5), sigmaX=2)
    blurred_f2 = cv2.GaussianBlur(blurred_f, (5, 5), sigmaX=2)
    alpha = 15
    sharpened = blurred_f + alpha * (blurred_f - blurred_f2)
    return np.clip(sharpened, 0, 1)


sharp_img_xtrain = np.array([img_sharpen(im) for im in X_train])
sharp_img_test = np.array([img_sharpen(im) for im in X_test])




## === cell 6
fig, axes = plt.subplots(1, 5, figsize=(15, 4))
for i, ax in enumerate(axes):
    idx = i * 200
    if idx < len(sharp_img_xtrain):
        ax.imshow(sharp_img_xtrain[idx])
        ax.set_title("Has cactus: " + str(Y_train[idx]))
plt.show()




## === cell 7
X_original = X_train.reshape(len(X_train), -1)
X_sharp = sharp_img_xtrain.reshape(len(sharp_img_xtrain), -1)
X_combined = np.concatenate([X_original, X_sharp], axis=1)

X_test_original = X_test.reshape(len(X_test), -1)
X_test_sharp = sharp_img_test.reshape(len(sharp_img_test), -1)
X_test_combined = np.concatenate([X_test_original, X_test_sharp], axis=1)

x_tr, x_val, y_tr, y_val = train_test_split(
    X_combined, Y_train, test_size=0.2, random_state=42, stratify=Y_train
)

class_weights_arr = class_weight.compute_class_weight(
    class_weight="balanced", classes=np.unique(y_tr), y=y_tr
)
class_weight_dict = {i: w for i, w in enumerate(class_weights_arr)}
sample_weights = np.array([class_weight_dict[int(lbl)] for lbl in y_tr])

scaler = StandardScaler()
x_tr_scaled = scaler.fit_transform(x_tr)
x_val_scaled = scaler.transform(x_val)
x_test_scaled = scaler.transform(X_test_combined)

mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    alpha=1e-4,
    batch_size=64,
    learning_rate="adaptive",
    max_iter=200,
    early_stopping=True,
    n_iter_no_change=10,
    random_state=42,
    verbose=False,
)

mlp.fit(x_tr_scaled, y_tr, sample_weight=sample_weights)

val_pred = mlp.predict_proba(x_val_scaled)[:, 1]
val_auc = roc_auc_score(y_val, val_pred)
print(f"Validation ROC‑AUC: {val_auc:.4f}")

test_probs = mlp.predict_proba(x_test_scaled)[:, 1]




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/930974684.py in <cell line: 0>()
     42 
     43 # Fit using sample weights to handle imbalance
---> 44 mlp.fit(x_tr_scaled, y_tr, sample_weight=sample_weights)
     45 
     46 # Validation AUC

TypeError: BaseMultilayerPerceptron.fit() got an unexpected keyword argument 'sample_weight'

## === cell 8
submission = pd.read_csv("../input/aerial-cactus-identification/sample_submission.csv")
if submission.empty:
    submission = pd.read_csv("../input/sample_submission.csv")

submission["has_cactus"] = test_probs
submission.to_csv("sample_submission.csv", index=False)
print("Submission file saved as sample_submission.csv")
print(submission.head())

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1730141603.py in <cell line: 0>()
      3     submission = pd.read_csv("../input/sample_submission.csv")
      4 
----> 5 submission["has_cactus"] = test_probs
      6 submission.to_csv("sample_submission.csv", index=False)
      7 print("Submission file saved as sample_submission.csv")

NameError: name 'test_probs' is not defined
